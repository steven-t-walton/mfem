//                                MFEM Example 41
//
// Compile with: make ex41
//
// Sample runs:
//  ex41
//  ex41 -cg
//  ex41 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.005 -tf 10
//  ex41 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//  ex41 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//  ex41 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//  ex41 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.001 -tf 9
//  ex41 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//  ex41 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//  ex41 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//  ex41 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//  ex41 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.01 -tf 8
//
// Device sample runs:
//
// Description:  This example code solves the time-dependent advection-diffusion
//               equation du/dt + v.grad(u) - a div(grad(u)) = 0, where v is a
//               given fluid velocity, a is the diffusion coefficient, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), and the use of IMEX
//               ODE time integrators.
//
//               The option to use continuous finite elements is available too.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Mesh bounding box
Vector bb_min, bb_max;

// Velocity coefficient
template<int problem=0>
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }
   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const real_t w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const real_t w = M_PI/2;
         real_t d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
template<int problem=0>
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               real_t rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const real_t s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                        std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const real_t f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}


/** A time-dependent operator for the right-hand side of the ODE. The weak
    form of the advection-diffusion equation is M du/dt = K u - S u + b,
    where M is the mass matrix, K and S are the advection and diffusion
    matrices, and b describes the flow on the boundary. In the case of IMEX
    evolution, the diffusion term is treated implicitly, and the advection
    term is treated explicitly.  */
class EERK_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K, &S;
   const Vector &b;
   unique_ptr<Solver> M_prec;
   
   
   mutable Vector z;

public:
   bool first_step = true; // is this the first time step?
   EERK_Evolution(BilinearForm &M_, BilinearForm &K_, BilinearForm &S_,
                  const Vector &b_, const bool eerk_true);

   void Mult(const Vector &x, Vector &y) const override;
   mutable StopWatch sw;
   CGSolver M_solver;
};
// Explicit Extrapolated Runge-Kutta (EERK) ODE solver
// An extrapolated runge-kutta method for the ODE y' = f(t,y)
class EERK : public ODESolver
{
private:
   DenseMatrix alpha;
   int s; // number of stages
   Vector b, bhat; // Butcher weights
   Vector c; // Butcher c vector
   DenseMatrix A; // Butcher A matrix
   double dt_prev; // previous time step size
   mutable Vector y, z; // state and temp vectors
   mutable Vector *k, *k_prev; // stage derivatives and previous step stage derivatives

public:
   EERK(const Vector b_, const Vector bhat_, const DenseMatrix A_, DenseMatrix alpha_); // constructor
   ~EERK();// destructor
   void Step(Vector &x, real_t &t, real_t &dt) override; // how to do a time step
   void Init(TimeDependentOperator &ode) override; // how to initialize the solver

};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 0;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 2; //RK2 (Midpoint)
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool paraview = true;
   bool cg = true;
   int vis_steps = 50;
   real_t diffusion_term = 0.0;
   real_t kappa = -1.0;
   real_t sigma = -1.0;
   bool visualization = false;
   bool visit = false;
   bool binary = false;
   int precision = 8;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order", "Order of the finite element spaces.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver type:\n"
                  " 1 - Forward Euler,\n"
                  " 2 - RK2 (Heun),\n"
                  " 3 - SSPRK3,\n"
                  " 4 - RK4,\n"
                  " 6 - RK6,\n"
                  " 7 - EERK (2nd order Explicit Extrapolated Runge-Kutta)");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&diffusion_term, "-dc", "--diffusion-coeff",
                  "Diffusion coefficient in the PDE.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&cg, "-cg", "--continuous-galerkin", "-dg",
                  "--discontinuous-galerkin",
                  "Use Continuous-Galerkin Finite elements (Default is DG)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   // 3. Define the IMEX (Split) ODE solver used for time integration. The IMEX
   // solvers currently available are: 61 - Forward Backward Euler,
   // 62 - IMEXRK2(2,2,2), 63 - IMEXRK2(2,3,2), and  64 - IMEX_DIRK_RK3.
   bool EERK_TRUE = false;
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 7: ode_solver = new EERK(
                 Vector({1./2., 1./2.}),
                 Vector({1./2., 1./2.}),
                 DenseMatrix({
                    {0.0, 0.0},
                    {1., 0.0}
                 }), DenseMatrix({
                    {0.0, 1.0},
                    {-1.0, 2.0}
                 }));
                 EERK_TRUE=true; break;
      case 8: ode_solver = new EERK(
                 Vector({1.0/4.0, 3.0/4.0}),
                 Vector({0.25, 0.75}),
                 DenseMatrix({
                    {0.0, 0.0},
                    {2./3., 0.0}
                 }), DenseMatrix({
                    {-1./2., 3./2.},
                    {-3./2., 5./2.}
                 })); 
                 EERK_TRUE=true; break;
      case 9: ode_solver = new EERK(
                 Vector({1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/2.0}),
                 Vector({1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/2.0}),
                 DenseMatrix({
                    {0.0, 0.0, 0.0, 0.0},
                    {1.0/2.0, 1.0/2.0, 0.0, 0.0},
                    {1.0/6.0, 1.0/6.0, 1.0/6.0, 0.0}
                 }), DenseMatrix({
                     { 0.0, -1.0, 1.0, 1.0},
                     { 1.0, -4.0, 3.0, 1.0},
                     { 3.0, -9.0, 6.0, 1.0},
                     { 2.0, -5.0, 3.0, 2.0}
                 })); 
                 EERK_TRUE=true; break; 
      default:
        cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        
   }  

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++) {mesh.UniformRefinement();}
   if (mesh.NURBSext) {mesh.SetCurvature(max(order, 1));}
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   FiniteElementCollection *fec = NULL;
   if (cg)
   {
      fec = new H1_FECollection(order, dim);
   }
   else
   {
      fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
   }
   FiniteElementSpace fes(&mesh, fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   std::unique_ptr<VectorFunctionCoefficient> velocity;
   if (0==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<0>));
   }
   else if (1==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<1>));
   }
   else if (2==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<2>));
   }
   else if (3==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<3>));
   }

   ConstantCoefficient diff_coeff(diffusion_term);

   BilinearForm m(&fes);
   BilinearForm adv_op(&fes);
   BilinearForm diff_op(&fes);

   Vector b(fes.GetTrueVSize());
   b = 0.0; //The inflow on the boundaries is set to zero.

   m.AddDomainIntegrator(new MassIntegrator);

   constexpr real_t alpha = -1.0;
   adv_op.AddDomainIntegrator(new ConvectionIntegrator(*velocity, alpha));

   diff_op.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   if (!cg)
   {
      adv_op.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity,
                                                                       alpha));
      adv_op.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity, alpha));
      diff_op.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma,
                                                            kappa));
      diff_op.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));
   }


   int skip_zeros = 0;
   m.Assemble(skip_zeros);
   adv_op.Assemble(skip_zeros);
   diff_op.Assemble(skip_zeros);

   m.Finalize(skip_zeros);
   adv_op.Finalize(skip_zeros);
   diff_op.Finalize(skip_zeros);

   // 7. Define the initial conditionds.
   std::unique_ptr<FunctionCoefficient> u0;
   if (0==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<0>));
   }
   else if (1==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<1>));
   }
   else if (2==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<2>));
   }
   else if (3==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<3>));
   }

   GridFunction u(&fes);
   u.ProjectCoefficient(*u0);

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Extrapolation Method", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Extrapolation Method", &mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   // 8. Set up paraview visualization, if desired.
   unique_ptr<ParaViewDataCollection> pv;
   if (paraview)
   {
      pv = make_unique<ParaViewDataCollection>("Extrapolation Method", &mesh);
      pv->SetPrefixPath("ParaView");
      pv->RegisterField("solution", &u);
      pv->SetLevelsOfDetail(order);
      pv->SetDataFormat(VTKFormat::BINARY);
      pv->SetHighOrderOutput(true);
      pv->SetCycle(0);
      pv->SetTime(0.0);
      pv->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 9. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   EERK_Evolution adv(m, adv_op, diff_op, b, EERK_TRUE);


   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      if(EERK_TRUE){
         if (ti == 0){
            adv.M_solver.SetMaxIter(100);
         }
         else{
            adv.M_solver.SetMaxIter(1);
         }
      }
      real_t dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);

      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (paraview)
         {
            pv->SetCycle(ti);
            pv->SetTime(t);
            pv->Save();
         }
         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }
         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }

      }
   }

   std::cout<< "cg total time :" << adv.sw.RealTime() << std::endl;

   delete fec;
   return 0;
}


// Implementation of class EERK_Evolution
EERK_Evolution::EERK_Evolution(BilinearForm &M_, BilinearForm &K_,
                               BilinearForm &S_, const Vector &b_, const bool eerk_true)
   : TimeDependentOperator(M_.FESpace()->GetTrueVSize()),
     M(M_), K(K_), S(S_), b(b_), z(height)
{
   Array<int> ess_tdof_list;
   if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
   {
      M_prec = make_unique<DSmoother>(M.SpMat());
      M_solver.SetOperator(M.SpMat());
   }
   
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = true;
   // if M_solver.iterative_mode = false -> r=z in 
   // CGSolver where r = b - Ax (i.e. residual is just rhs and initial guess is 0)
   // if M_solver.iterative_mode = true -> r = z-My
   // the default is false
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   // r0 = max( (Br_init,r_init) * rel_tol^2, abs_tol^2)
   // where B = I or preconditioner and r_init = b-Ax_0 or just b (see above on iterative mode) 
   // r0 is the stopping criterion so that if ( ( (B r(i), r(i)) <= r0 ) <=> ( betanom <= r0 ) ) => Stop
   if (!eerk_true)
   {
      M_solver.SetMaxIter(50);
   }
   // max number of iters for which we will try to reach the above stopping criterion
   M_solver.SetPrintLevel(0);

}

void EERK_Evolution::Mult(const Vector &x, Vector &y) const
{
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;

   sw.Start();
   M_solver.Mult(z, y);// z-My = (Kx-Sx)-My
   // if M_solver.iterative_mode = false -> r=z in 
   // CGSolver where r = b - Ax (i.e. residual is just rhs and initial guess is 0)
   // if M_solver.iterative_mode = true -> r = z-My
   // the default is false
   sw.Stop();
   // std::cout<< "number of iterations :" << M_solver.GetNumIterations() << std::endl;
   
}

EERK::EERK(const Vector b_, const Vector bhat_, const DenseMatrix A_, DenseMatrix alpha_)
   : ODESolver(), b(b_), bhat(bhat_), A(A_), alpha(alpha_)
{
   s = b.Size();
   c.SetSize(b.Size());
   dt_prev = -1.0;
   k = new Vector[s];
   k_prev = new Vector[s];
   // compute c from A
   for (int i = 0; i < s; i++)
   {
      c(i) = 0.0;
      for (int j = 0; j < s; j++)
      {  
         c(i) += A(i,j);
      } 
   }
}

EERK::~EERK()
{
   delete[] k;
   delete[] k_prev;
}

void EERK::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_); // f is actually g...
   int n = f->Width(); // f is actually g...
   
   y.SetSize(n, mem_type);
   for (int i = 0; i < s; i++)
   {
      k[i].SetSize(n, mem_type);
      k_prev[i].SetSize(n, mem_type);
   }
   
}


void EERK::Step(Vector &x, real_t &t, real_t &dt)
{
   
   // form exatrpolated k_i stages using alpha matrix
   
   
   for (int i = 0; i < s; i++)
   {  
      k[i] = 0.0;
      for (int j = 0; j < s; j++)
      {
         k[i].Add(alpha(i,j), k_prev[j]);
      }
   }


   
   
   for (int i = 0; i < s; i++)
   {  
      y = x;
      for (int j = 0; j < i; j++)
      {
         add(y, A(i,j)*dt, k[j], y);
      }
      f->SetTime(t + c(i)*dt);
      f->Mult(y, k[i]);
   }

   for (int i = 0; i < s; i++)
   {
      x.Add(b(i)*dt, k[i]);
   }

   dt_prev = dt;
   for (int i = 0; i < s; i++)
   {
      k_prev[i] = k[i];
   }

   t += dt;
}
