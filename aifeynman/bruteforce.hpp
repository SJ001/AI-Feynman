#ifndef INTEGRATE_HPP
#define INTEGRATE_HPP
#include <string>

extern "C"
int bruteforce(const double *data, int n_data, int n_input, int duration, char operation_type, float bitmargin, float nu, std::string save_path, std::string arity2templates_file);

//wchar_t*

//int bf_gradient(float *data, int n_data, int dp_size, int duration, float bitmargin, float nu, wchar_t* save_path_p, wchar_t* arity2templates_file);

extern "C"
int bf_gradient(const double *data, int n_data, int dp_size, int duration, float bitmargin, float nu, std::string save_path, std::string arity_path);

#endif
