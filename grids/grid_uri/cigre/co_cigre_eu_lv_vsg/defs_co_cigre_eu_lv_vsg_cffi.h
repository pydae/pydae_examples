void f_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void g_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void f_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void g_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void h_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_run_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_run_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_run_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_run_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_run_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_run_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void de_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);