#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <fstream>
#include <typeinfo>
#include <math.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include <iterator>
#include <unordered_map>
#include <stack>

#include "bruteforce.hpp"


using namespace std;
typedef double T;

//extern "C" double eval_(int n, string arities, string ops, const double *x, const double *val);

typedef void (*evaluator)(stack<T> &ev_stack, const T *data, bool &nan_flag);
//typedef void (*evaluator)(T *stack, int &stack_len, const T *data, bool &nan_flag);

T get_input_dp(const T *data, int i, int j, int dp_size) {
    T val = data[i * dp_size + j];
    return val;
}

T get_output_dp(const T *data, int i, int dp_size) {
    T val = data[(i+1) * dp_size - 1];
    return val;
}

char* ops(const string i, char* variables, char* operations_1, char* operations_2){
    if (i=="0") {return variables;}
    else if (i=="1") {return operations_1;}
    else {return operations_2;}
}


/*print a vector of type ector<T>*/
void printvec(vector<T> vec){
    int num = vec.size();
    for (int i=0;i<num;i++){
        cout << vec[i] << endl;
    }
}

void printintvec(vector<int> vec){
    int num = vec.size();
    for (int i=0;i<num;i++){
        cout << vec[i] << endl;
    }
}


/*print a reverse polish notation*/
void printrpn(vector<char> RPN){
    int num = RPN.size();
    for (int i=0;i<num;i++){
        cout << RPN[i];
    }
    cout << endl;
}


/*convert flat_id to id in each digit. For example 0->0000,1->0001,2->0010,4->0100,8->1000 for prod_dims=(2,2,2,2)*/
vector<int> each_ids(int flat_id, vector<int> prod_dims){
    int len = prod_dims.size();
    vector<int> each_id;
    for (int i=0;i<len;i++){
        int de = prod_dims[len-i-1];
        int q = flat_id/de;
        flat_id = flat_id - q*de;
        each_id.push_back(q);
    }
    return each_id;
}

void s_plus(stack<T> &ev_stack, const T *data, bool &nan_flag) {
	const auto a  = ev_stack.top();
	ev_stack.pop();
	const auto b  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(b + a);
}

void s_minus(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	const auto b  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(b - a);
}

void s_mult(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	const auto b  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(b * a);
}

void s_div(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	const auto b  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(b / a);
}

void s_incr(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(a + 1);
}

void s_decr(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(a - 1);
}

void s_neg(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(-a);
}

void s_double(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(2*a);
}

void s_square(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(a * a);
}

void s_log(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	if (a <= 0) {nan_flag = true;}
	ev_stack.emplace(log(a));
}

void s_exp(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(exp(a));
}

void s_sin(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(sin(a));
}

void s_cos(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(cos(a));
}

void s_abs(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(abs(a));
}

void s_asin(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(asin(a));
}

void s_atan(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(atan(a));
}

void s_sqrt(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	if (a < 0) {nan_flag = true;}
	ev_stack.emplace(sqrt(a));
}

void s_inv(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	if (abs(a) < pow(2, -30)) {nan_flag = true;}
	ev_stack.emplace(1/a);
}

void s_onedouble(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    const auto a  = ev_stack.top();
	ev_stack.pop();
	ev_stack.emplace(1 + 2 * a);
}

void s_pi(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(atan(1)*4);
}

void s_zero(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(0);
}

void s_one(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(1);
}

void s_x0(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[0]);
}

void s_x1(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[1]);
}

void s_x2(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[2]);
}

void s_x3(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[3]);
}

void s_x4(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[4]);
}

void s_x5(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[5]);
}

void s_x6(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[6]);
}

void s_x7(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[7]);
}

void s_x8(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[8]);
}

void s_x9(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[9]);
}

void s_x10(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[10]);
}

void s_x11(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[11]);
}

void s_x12(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[12]);
}

void s_x13(stack<T> &ev_stack, const T *data, bool &nan_flag) {
    ev_stack.emplace(data[13]);
}

const static unordered_map<char, evaluator> lookup_table {
    {'+', s_plus},
    {'-', s_minus},
    {'*', s_mult},
    {'/', s_div},
    {'>', s_incr},
    {'<', s_decr},
    {'~', s_neg},
    {'O', s_double},
    {'Q', s_square},
    {'L', s_log},
    {'E', s_exp},
    {'S', s_sin},
    {'C', s_cos},
    {'A', s_abs},
    {'N', s_asin},
    {'T', s_atan},
    {'R', s_sqrt},
    {'I', s_inv},
    {'J', s_onedouble},
    {'P', s_pi},
    {'0', s_zero},
    {'1', s_one},
    {'a', s_x0},
    {'b', s_x1},
    {'c', s_x2},
    {'d', s_x3},
    {'e', s_x4},
    {'f', s_x5},
    {'g', s_x6},
    {'h', s_x7},
    {'i', s_x8},
    {'j', s_x9},
    {'k', s_x10},
    {'l', s_x11},
    {'m', s_x12},
    {'n', s_x13}
};

// https://stackoverflow.com/questions/46787306/postfix-evaluation-is-very-slow-optimization
/*evaluate the symbolic expression at a point z*/
T H_evaluate(const vector<char> &RPN, int arity_len,const T *data, int i, int dp_size, bool &nan_flag){
    // TODO: Should read ops from e.g. 14ops.txt (choice is user-provided)
    stack<T> ev_stack;
    const T *data_i = data + i * dp_size;

    for (const auto& symbol : RPN) {
        const auto method = lookup_table.find(symbol);
        if (method != lookup_table.end()) {
            (method->second)(ev_stack, data_i, nan_flag);
        }
        else {
            cout << "bad input in RPN" << endl;
        }
    }
    return ev_stack.top();
}


/*vector norm*/
T norm(const vector<T> & vec, int dim){
    T squared = 0.0;
    for (int i=0;i<dim;i++){
        squared = squared + pow(vec[i],2);
    }
    return sqrt(squared);

}

/*normalize a vector*/
vector<T> normalized_vec(const vector<T> & vec, int dim){
    T norm_vec = norm(vec, dim);
    vector<T> vecp = vec;
    if (norm_vec<1e-12) {
        for (int i=0;i<dim;i++){
            vecp[i] = 1.0;
            /*throw 20;*/
        }
        norm_vec = norm(vecp, dim);
    }
    for (int i=0;i<dim;i++){
        vecp[i] = vecp[i]/norm_vec;
    }
    return vecp;
}

/*compute inner product of two vectors*/
T my_inner_product(vector<T> vec1,vector<T> vec2, int dim){
    T eps = 0.0;
    for (int i=0;i<dim;i++){
        eps = eps + vec1[i]*vec2[i];
    }
    return eps;
}

T vector_median(vector<T> &v){
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

T limit(T val){
    if (val == 0.0) {return val;}
    T sgn = val/abs(val);
    T mn = min(abs(val), 666.0);
    return sgn*mn;
}


string wchar2string(wchar_t* str_p) {
    string str;
    while(*str_p) {str += (char)*str_p++;}
    return str;
}


int bruteforce(const T *data, int n_data, int dp_size, int duration, char operation_type, float bitmargin, float nu, string save_path, string arity_path){
    auto work_duration = std::chrono::seconds{duration};
    auto now = std::chrono::steady_clock::now;
    //string save_path = wchar2string(save_path_p);
    int num_templates = 1000; /*number of templates tried*/ // TODO: increase this limit or reload new templates in favor for duration parameter
    T epsilon = pow(0.5, 30.0);
    //cout << "duration: " << duration << endl;
    const int input_dim = dp_size - 1; /*Number of input varibles*/
    // TODO: Max's unicode trick
    char variables2[] = {'a','b','c','d','e','f','g','h','i','j','k', 'l','m','n','\0'};
    char variables[input_dim+2];
    if (input_dim > 14){cout << "Support <15 variables!";};
    for (int i=0;i<input_dim;i++){
        variables[i] = variables2[i];
    }
    variables[input_dim] = 'P';
    //variables[input_dim+1] = '0';
    //variables[input_dim+2] = '1';
    variables[input_dim+1] = '\0';
    //char operations_1[] = {'>','<','~','\\','O','Q','I','J','\0'};
    //char operations_2[] = {'+','*','-','/','\0'};
    char operations_1[] = {'>','<','~','I','R','S','C','L','E','\0'};
    char operations_2[] = {'+','*','-','/','\0'};

    //string arity_path = wchar2string(arity2templates_file);
    ifstream input2(arity_path); /*read arity file*/
    vector<string> arities;

    int template_id = 0;
    for (string line; getline(input2,line);) /* read arity file and add num_template number of strings into arities vector */
    {
        if (template_id == num_templates){break;}
        arities.push_back(line);
        template_id = template_id + 1;
    } /* save arity data into arities */

    T bestbits = 99999999; // such that the first expr will succeed
    T sigma = 1;
    T value[n_data];
    vector<T> offsets;
    T error_bits;
    int nevals = 0;
    int nformulas = 0;

    auto stop_time = now() + work_duration;

    /* loop over arities */
    for (int k=0;k<num_templates;k++){
        if (now() > stop_time) {
            //cout << "Time limit passed at k=" << k << endl;
            //cout << "nevals = " << nevals << endl << "nformulas = " << nformulas << endl;
            break;
        }
        string arity = arities[k]; //"0021"
        int arity_len = arity.length();
        // TODO: .

        vector<int> dims;
        vector<int> prod_dims;
        vector<string> symbol_types;


        int flat_size = 1;
        /*For each template, compute how many RPNs.*/
        for (int i=0; i<arity.length();i++){
            string symbol_type = arity.substr(i,1);
            int dim = strlen(ops(symbol_type, variables, operations_1,operations_2)); /* number of entries with arity symbol_type */
            symbol_types.push_back(symbol_type);
            dims.push_back(dim);
            prod_dims.push_back(flat_size);
            flat_size = flat_size * dim;
        } /* save into (symbol_type), (dims), prod_dims, flat_size */


        T bitmean;
        T bitexcess;
        T median;
        T bitsdev;
        vector<char> RPN;
        //vector<char> RPN2 = {'a','P', '*', 'I', 'E', 'P', '*', 'P', '*', 'P', '*'};


        /*Test each RPN to see if it agrees with data*/

        for (int i=0; i<flat_size;i++){
            nformulas += 1;
            vector<int> each_id = each_ids(i, prod_dims); /* will yield all possible bitstrings in basis prod_dims */
            RPN.clear();
            /*construct RPN*/
            for (int j=0;j<arity_len;j++){
                char* ops_ = ops(arity.substr(j,1), variables, operations_1, operations_2); /* yields one of the vectors variables, operations_1 or operations_2 */
                char symbol = ops_[each_id[arity_len-1-j]];
                RPN.push_back(symbol);
            }

            //if (RPN != RPN2) { continue;}
            //cout << "RPN:" << endl;
            //printrpn(RPN);
            bool rejected = false;
            int jtest = 2;
            offsets.clear();
            bool nan_flag = false;
            int last_j = 0;
            T offst;
            for (int j=0; j<n_data; j++) {
                nevals += 1;
                last_j = j;
                // TODO: Clean up this stack mess
                string RPN_str(RPN.begin(), RPN.end());

                value[j] = H_evaluate(RPN, arity_len, data, j, dp_size, nan_flag);

                //value[j] = H_evaluate_ifs(RPN, arity_len, data, j, dp_size, nan_flag);

                if (nan_flag) {
                    rejected = true;
                    break;
                }

                if (operation_type == '+') {

                    offst = get_output_dp(data, j, dp_size) - value[j];
                    if (abs(offst) > 1 / epsilon) {
                        rejected = true;
                        break;
                    } // Otherwise numerical cancellation can masquerade as success
                } else {
                    if (value[j] == 0.0) {
                        rejected = true;
                        break;
                    } // reject if target/value was nan
                    offst = get_output_dp(data, j, dp_size) / value[j];
                    if (abs(offst) < epsilon) {
                        rejected = true;
                        break;
                    } // Otherwise numerical cancellation can masquerade as success
                    if (abs(offst) > 1 / epsilon) {
                        rejected = true;
                        break;
                    } // Otherwise numerical cancellation can masquerade as success
                }

                offsets.push_back(offst);

                if (j > jtest) { // time for another test
                    median = vector_median(offsets);
                    T sum_lin = 0;
                    T sum_quad = 0;
                    for (int l = 0; l <= j; l++) {
                        // is it really neccessary to save all the bit values? we only need the current one
                        if (operation_type == '+') {
                            error_bits = max(1.44269504089 * log((abs(offsets[l] - median)) / epsilon), 0.0);
                            //bits.push_back(max(1.44269504089 * log((abs(offsets[l] - median)) / epsilon), 0.0));
                        }
                        else {
                            error_bits = max(1.44269504089 * log((abs(get_output_dp(data, l, dp_size) * (1 - (median / offsets[l])))) / epsilon), 0.0);
                            //bits.push_back(max(1.44269504089 * log((abs(get_output_dp(data, l, dp_size) * (1 - (median / offsets[l])))) / epsilon), 0.0));
                        }
                        sum_lin = sum_lin + error_bits;
                        sum_quad = sum_quad + error_bits * error_bits;
                    }
                    bitmean = sum_lin / (j + 1);
                    if (!(bitmean == bitmean)) {
                        //cout << "the following expression was uncaught nan:" << endl;
                        //copy(RPN.begin(), RPN.end(), ostream_iterator<char>(cout, " "));
                        //cout << endl;
                        nan_flag = true;
                        rejected = true;
                        break;
                    }
                    bitsdev = sqrt(abs(sum_quad / (j + 1) - pow(sum_lin / (j + 1), 2)));
                    //cout << "test at j = " << j << endl << "bitmean = " << bitmean << endl << "bitsdev = " << bitsdev << endl;
                    bitexcess = bitmean - bestbits + bitmargin;
                    T z = sqrt(j + 1) * bitexcess / sigma;
                    if (z > nu) {
                        rejected = true;
                        break;
                    }
                    jtest = 2 * jtest;
                }

                if (rejected) {break;}
            }

            /* save accurate RPNs */
            if (!rejected && bitexcess < 0){
                bestbits = min(bitmean, bestbits);
                sigma = bitsdev;
                string RPN_string(RPN.begin(), RPN.end());
                ofstream myfile;
                T ev = nevals/nformulas;
                //cout << "Winner : " << RPN_string << " ... " <<  bestbits << " ... " << limit(median) << " ... " << nevals << " ... " << nformulas << " ... " << ev << endl;
                myfile.open (save_path, std::ios_base::app);
                myfile << RPN_string << " " << bestbits << " " << to_string(limit(median)) << " " << nevals << " " << nformulas << " " << ev << endl;
                myfile.close();
            }
        }
    }
    /*
    cout << "<-----------------RESULTS-------------->" << endl;
    cout << "nan rejects: " << reject_type[0] << endl;
    cout << "numerical rejects: " << reject_type[1] << endl;
    cout << "z rejects: " << reject_type[2] << endl;
    cout << "not accurate: " << reject_type[3] << endl;
    cout << "-> of which where equal: " << equiv_exprs << endl;
    cout << "winners: " << reject_type[4] << endl;
     */

    return 0;
}

T grad_bitloss(vector<T> vec1, vector<T> vec2, int n) {
    T dot = my_inner_product(vec1, vec2, n);
    T loss = (1-abs(dot))*(pow(2, 30));
    //if (!(loss < 0 && loss > 0)) {return 1e30;} // this was a NaN
    T bit_loss = max(1.44269504089*log(loss), 0.0); // 1.44...*log = log_2
    return bit_loss;
}


/*compute the gradient of the symbolic formula at point z*/
//vector<T> H_grad_evaluate(vector<char> RPN, int arity_len, vector<T> z, const int dim){
vector<T> H_grad_evaluate(const vector<char> &RPN, int arity_len, const T *data, int j, int dp_size, int nvar, bool &nan_flag){
    T eps = 1e-6;
    vector<T> grads;

    T z[nvar];
    copy(data + dp_size * j, data + dp_size*j + nvar, z);

    for (int i=0;i<nvar;i++){
        //vector<T> zp = z;
        //vector<T> zm = z;
        z[i] = z[i] + eps;
        //T Hp = H_evaluate_2(RPN, arity_len, zp);
	T Hp = H_evaluate(RPN, arity_len, z, 0, nvar, nan_flag);
        z[i] = z[i] - 2*eps;
	T Hm = H_evaluate(RPN, arity_len, z, 0, nvar, nan_flag);
        //T Hm = H_evaluate_2(RPN, arity_len, zm);
        z[i] = z[i] + eps;

        T grad = (Hp-Hm)/(2*eps);
        grads.push_back(grad);
    }
    grads = normalized_vec(grads, nvar);

    return grads;
}


// TODO: change data type to T
int bf_gradient(const T *data, int n_data, int dp_size, int duration, float bitmargin, float nu, string save_path, string arity_path){
	auto work_duration = std::chrono::seconds{duration};
	auto now = std::chrono::steady_clock::now;
	//string save_path = wchar2string(save_path_p);
	//
	// TODO: increase this limit or reload new templates in favor for duration parameter
	int num_templates = 1000;

	/*Number of input varibles*/
	const int nvar = dp_size / 2;

	if (nvar * 2 != dp_size) {
		cout << "Error: dp_size is odd. Exiting." << endl;
		return 1;
	}

	// TODO: move normalization to python-side

	int input_dim = nvar;
	char variables2[] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','\0'};
	char variables[input_dim+4];
	if (input_dim > 14){
		cout << "Error: Support <15 variables! Exiting." << endl;
		return 1;
	};
	for (int i=0;i<input_dim;i++){
		variables[i] = variables2[i];
	}
	
	variables[input_dim] = 'P';
	//variables[input_dim+1] = '0';
	//variables[input_dim+2] = '1';
	variables[input_dim+1] = '\0';

	//char operations_1[] = {'>','<','~','\\','O','Q','I','J','\0'};
	//char operations_2[] = {'+','*','-','/','\0'};
	    char operations_1[] = {'>','<','~','I','R','S','C','L','E','\0'};
	    char operations_2[] = {'+','*','-','/','\0'};

	//string arity_path = wchar2string(arity2templates_file);
	ifstream input2(arity_path);
	vector<string> arities;
	int template_id = 0;

	/* read arity file and add num_template number of strings into arities vector */
	for (string line; getline(input2,line);)
	{
		if (template_id == num_templates){break;}
		arities.push_back(line);
		template_id = template_id + 1;
	}
	//input2.close();
	/* save arity data into arities */
	T bestbits = 99999999; // such that the first expr will be kept
	T sigma = 1;
	T lossbits;
	T meanbits;
	bool timeout = false;
	auto stop_time = now() + work_duration;

	/* loop over arities */

	for (int k=0;k<num_templates;k++){
		string arity = arities[k]; //"0021"
		int arity_len = arity.length();
		vector<int> dims;
		vector<int> prod_dims;
		vector<string> symbol_types;
		int flat_size = 1;

		/*For each template, compute how many RPNs.*/
		for (long unsigned int i=0; i<arity.length();i++){
			string symbol_type = arity.substr(i,1);
			int dim = strlen(ops(symbol_type, variables, operations_1,operations_2));
			/* number of entries with arity symbol_type */

			symbol_types.push_back(symbol_type);
			dims.push_back(dim);
			prod_dims.push_back(flat_size);
			flat_size = flat_size * dim;
		}
		/* save into (symbol_type), (dims), prod_dims, flat_size */

		T bitexcess;
		vector<char> RPN;

		/*Test each RPN to see if it agrees with data*/
		for (int i=0; i<flat_size;i++){
			/* will yield all possible bitstrings in basis prod_dims */
			vector<int> each_id = each_ids(i, prod_dims);
			RPN.clear();

			/*construct RPN*/
			for (int j=0;j<arity_len;j++){
				char* ops_ = ops(arity.substr(j,1), variables, operations_1, operations_2);
				/* yields one of the vectors variables, operations_1 or operations_2 */

				char symbol = ops_[each_id[arity_len-1-j]];
				RPN.push_back(symbol);
			}

			// TODO: include check if all nvar variables are included in the RPN. otherwise discard.

			bool rejected = false;
			T bitsum = 0;
			bool nan_flag = false;

			for (int j=0; j<n_data; j++){
				T z;
				//vector<T> x(data + dp_size * j, data + dp_size * j + nvar);
				vector<T> gradf(data + dp_size * j + nvar, data + dp_size * j + 2*nvar);
				vector<T> gradfhat = H_grad_evaluate(RPN, arity_len, data, j, dp_size, nvar, nan_flag);
				// was previously grad_bitloss(x, gradfhat, nvar)...
				lossbits = grad_bitloss(gradf, gradfhat, nvar);

				if (isnan(lossbits)) {
					rejected = true;
					break;
				}

				bitsum += lossbits;
				meanbits = bitsum/(j+1);
				bitexcess = meanbits - bestbits + bitmargin;
				z = sqrt(j+1)*bitexcess/sigma;

				if (z > nu) {
					rejected = true;
					break;
				}
			}


			/* save accurate RPNs */
			if (!rejected && bitexcess < 0){
				bestbits = min(meanbits, bestbits);
				string RPN_string(RPN.begin(), RPN.end());
				ofstream myfile;
				cout << "Winner : " << RPN_string << " ... " <<  bestbits << endl;
				myfile.open (save_path, std::ios_base::app);
				myfile << RPN_string << " " << bestbits << endl;
				myfile.close();
			}
			if (now() > stop_time) {
				cout << "Time limit passed. Exiting bf module..." << endl;
				timeout = true;
				break;
			}
		}
		if (timeout) {break;}
	}

	return 0;
}
