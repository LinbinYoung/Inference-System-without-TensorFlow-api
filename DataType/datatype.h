//
// Created by Linbin Yang on 2019-12-28.
//

#ifndef C_INFERENCE_DATATYPE_H
#define C_INFERENCE_DATATYPE_H

#include <common/commonlib.h>
#include <fstream>
#include <limits>

namespace MultiEigen{

    template <typename T>
    struct Eigen_Vector{
    public:
        Eigen_Vector(){}
        Eigen_Vector(int m, Json::Value node){
            this->data.resize(m);
            for (int i = 0; i < m; i ++){
                this->data(i) = (T)node[i].asDouble();
            }
        }
        Eigen_Vector(int m){
            this->data = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(m);
        }
        Eigen::Matrix<T, Eigen::Dynamic, 1>& getData(){
            return this->data;
        }
        void setData(Eigen::Matrix<T, Eigen::Dynamic, 1> current){
            this->data = current;
        }
        int Sinsize(){
            return this->data.size();
        }
        void setData(int m, Json::Value node){
            this->data.resize(m);
            for (int i = 0; i < m; i ++){
                this->data(i) = (T)node[i].asDouble();
            }
        }
        void Printout(){
            for (int i = 0; i < this->data.size(); i ++){
                cout << this->data[i] << " ";
            }
            cout << endl;
        }
        //auto inline
        inline T &operator[](int i){
            return this->data(i);
        }
        template <typename U>
        friend std::ostream& operator<<(std::ostream &os, const Eigen_Vector<U> &m);
    private:
        Eigen::Matrix<T, Eigen::Dynamic, 1> data;
    };
    template <typename T>
    std::ostream& operator<<(std::ostream &os, const Eigen_Vector<T> &m){
        os << "[" << m.data << "]";
        return os;
    }

    template <typename T>
    class Eigen_Col_iterator{
    public:
        Eigen_Col_iterator(){}
        Eigen_Col_iterator<T>& initailizer(T *data, size_t index, size_t size, size_t step){
            this->start = data;
            this->cur_index = index;
            this->max_size = size;
            this->step = step;
            this->start = this->start + index;
            return *this;
        }
        bool isValid(){
            return this->cur_index < this->max_size;
        }
        Eigen_Col_iterator<T>& operator++(int){
            Eigen_Col_iterator<T> temp = *this;
            this->cur_index = this->cur_index + this->step;
            this->start = this->start + this->step;
            return temp;
        }
        Eigen_Col_iterator<T>& operator--(int){
            Eigen_Col_iterator<T> temp = *this;
            this->cur_index = this->cur_index - this->step;
            this->start = this->start - this->step;
            return temp;
        }
        T &operator*(){
            return *(this->start);
        }
        bool operator==(const Eigen_Col_iterator<T>& iter2){
            return this->start == iter2.start;
        }
        bool operator!=(const Eigen_Col_iterator<T>& iter2){
            return this->start != iter2.start;
        }
    public:
        T *start;
    private:
        size_t step;
        size_t cur_index;
        size_t max_size;
    };

    template <typename T>
    class Eigen_2D_iterator{
    public:
        Eigen_2D_iterator(){}
        Eigen_2D_iterator(T* data, size_t index, size_t col, size_t row){
            this->max_size = col * row;
            this->step = col;
            this->cur_index = index;
            this->data = data;
            this->data = this->data + index;
        }
        bool isValid(){
            return this->cur_index < this->max_size;
        }
        Eigen_2D_iterator<T>& operator++(int){
            Eigen_2D_iterator<T> temp = *this;
            this->cur_index = this->cur_index + this->step;
            this->data = this->data + this->step;
            return temp;
        }
        Eigen_2D_iterator<T>& operator--(int){
            Eigen_2D_iterator<T> temp = *this;
            this->cur_index = this->cur_index - this->step;
            this->data = this->data - this->step;
            return temp;
        }
        Eigen_Col_iterator<T> &operator*(){
            return this->start.initailizer(this->data, 0, this->step, 1);
        }
        bool operator==(const Eigen_2D_iterator<T>& iter2){
            return this->data == iter2.data;
        }
        bool operator!=(const Eigen_2D_iterator<T>& iter2){
            return this->data != iter2.data;
        }
        Eigen_Col_iterator<T> begin(){
            return this->start.initailizer(this->data, 0, this->step, 1);
        }
        Eigen_Col_iterator<T> end(){
            return this->start.initailizer(this->data, this->step, this->step, 1);
        }
        T* data;
        Eigen_Col_iterator<T> start;
    private:
        size_t cur_index;
        size_t step;
        size_t max_size;
    };

    /*
        padding: VALID = without padding, SAME = with zero padding
        Note that in TensorFlow, "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd,
        it will add the extra column to the right, the same logic applies vertically: there may be an extra row of zeros at the bottom.
    */

    enum struct padding_type{
        valid,
        same
    };

#define FILTER_DIM(input, filter, stride) (((input) - (filter))/(stride) + 1)
#define NEEDED_DIMENSION(input, filter, stride) ((((input/stride))-1) * (stride) + filter - input)
    template <typename T>
    struct Eigen_2D{
    public:
        Eigen_2D(){}
        Eigen_2D(int m, int n){
            this->data =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(m,n);
        }
        Eigen_2D(int m, int n, Json::Value node){
            this->data.resize(m, n);
            for (int i = 0; i < m; i ++){
                for (int j = 0; j < n; j ++){
                    this->data(i, j) = (T)node[i][j].asDouble();
                }
            }
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& getData(){
            return this->data;
        }
        T &operator()(int x, int y){
            return this->data(x, y);
        }
        template <typename U>
        friend std::ostream& operator<<(std::ostream &os,  Eigen_2D<U> &m);

        void setData(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> current){
            this->data = current;
        }
        void setData(Eigen_2D<T> &current){
            this->data = current.data;
        }
        void setData(int m, int n, Json::Value node){
            this->data.resize(m, n);
            for (int i = 0; i < m; i ++){
                for (int j = 0; j < n; j ++){
                    this->data(i, j) = (T)node[i][j].asDouble();
                }
            }
        }
        void initialize(int m, int n){
            this->data =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(m,n);
        }
        size_t cols(){
            return this->data.cols();
        }
        size_t rows(){
            return this->data.rows();
        }
        void resize(int m, int n){
            this->data.resize(m, n);
        }
        Eigen_2D_iterator<T> begin(){
            Eigen_2D_iterator<T> b(&this->data(0,0), 0, this->cols(), this->rows());
            return b;
        }
        Eigen_2D_iterator<T> end(){
            Eigen_2D_iterator<T> b(&this->data(0,0), this->cols()*this->rows(), this->cols(), this->rows());
            return b;
        }
    public:
        //add with broadcast
        Eigen_2D<T> AddBoradCast(Eigen_2D<T> new_vec){
            Eigen_2D<T> res;
            Eigen::Matrix<T, Eigen::Dynamic, 1> vec(new_vec.rows());
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& vec_infere = new_vec.getData();
            for(int i = 0; i < new_vec.rows(); i ++){
                vec(i) = vec_infere(i,0);
            }
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output = this->data.transpose();
            output.colwise() += vec;
            res.setData(output.transpose());
            return res;
        }
        Eigen_2D<T> AddBoradCast(Eigen_Vector<T> new_vec){
            Eigen_2D<T> res;
            Eigen::Matrix<T, Eigen::Dynamic, 1>& vec = new_vec.getData();
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output = this->data.transpose();
            output.colwise() += vec;
            res.setData(output.transpose());
            return res;
        }
        void AddBoradCast(Eigen_Vector<T> new_vec, int index){
            T single_add = new_vec[index];
            for (auto iter = this->begin(); iter != this->end(); iter++){
                //std::transform(iter.begin(), iter.end(), iter.begin(), f);
                for (auto elem = iter.begin(); elem != iter.end(); elem ++){
                    *elem = *elem + single_add;
                }
            }
        }
        //add without broadcast
        Eigen_2D<T> AddWithoutBroadCast(Eigen_2D<T> mat2){
            Eigen_2D<T> res;
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output = this->data + mat2.getData();
            res.setData(output);
            return res;
        }
        //matmul
        Eigen_2D<T> Matmul(Eigen_2D<T> mat2){
            Eigen_2D<T> res;
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output = this->data * mat2.data;
            res.setData(output);
            return res;
        }
        //argmax
        Eigen_Vector<T> Argmax(bool row_or_col){
            Eigen_Vector<T> res;
            res.setData(this->Argmax_helper(this->data, row_or_col));
            return res;
        }
        //convd
        Eigen_2D<T> convd_base(Eigen_2D<T> kernel, std::vector<int> stride, padding_type padding){
            int x_axis_stride = stride[1];
            int y_axis_stride = stride[2]; //leave out the 0th and 3th elements
            int left = 0;
            int right = 0;
            int up = 0;
            int down = 0;
            if (padding == padding_type::same) {
                int x_needed = NEEDED_DIMENSION(this->cols(), kernel.cols(), x_axis_stride);
                int y_needed = NEEDED_DIMENSION(this->rows(), kernel.rows(), y_axis_stride);
                left = x_needed / 2;
                right = x_needed - left;
                up = y_needed / 2;
                down = y_needed - up;
            }
            //return shape
            Eigen_2D<T> res;
            int col_shape = right+left+this->cols();
            int row_shape = up+down+this->rows();
            int ac_col_shape = FILTER_DIM(col_shape, kernel.cols(), x_axis_stride);
            int ac_row_shape = FILTER_DIM(row_shape, kernel.rows(), y_axis_stride);
            //expand the image with possible padding
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> new_image =  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(row_shape, col_shape);
            new_image.resize(row_shape, col_shape);
            new_image.block(up, left, this->rows(), this->cols()) << this->data;
            //now start compute the convd_base
            res.resize(ac_row_shape, ac_col_shape);
            for (int i = 0; i < ac_row_shape; i ++){
                for (int j = 0; j < ac_col_shape; j ++){
                    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> unit_image = new_image.block(i*x_axis_stride, j*y_axis_stride, kernel.rows(), kernel.cols());
                    Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> v1(unit_image.data(), unit_image.size());
                    Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> v2(kernel.getData().data(), kernel.getData().size());
                    res(i, j) = v1.dot(v2);
                }
            }
            return res;
        }
        //softmax2d
        Eigen_2D<T> softmax2d(bool row_or_col){
            Eigen_2D<T> res;
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> softres = this->SoftmaxOp(this->data, row_or_col);
            res.setData(softres);
            return res;
        }
        //apply(Relu, Sigmoid ......)
        Eigen_2D<T> apply(const std::function<T(const T&)> &f){
            Eigen_2D<T> res;
            res.setData(this->data);
            // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& f_data = res.getData();
            for (auto iter = res.begin(); iter != res.end(); iter++){
                //std::transform(iter.begin(), iter.end(), iter.begin(), f);
                for (auto elem = iter.begin(); elem != iter.end(); elem ++){
                    *elem = f(*elem);
                }
            }
            return res;
        }
    protected:
        Eigen::Matrix<T, Eigen::Dynamic, 1> Argmax_helper(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat1, bool row_or_col){
            if (!row_or_col){
                return Argmax_helper(mat1.transpose(), !row_or_col);
            }
            Eigen::Matrix<T, Eigen::Dynamic, 1> output(mat1.rows());
            for (int i = 0; i < mat1.rows(); i ++){
                output(i) = (T)this->FindMaxIndex(mat1, i, mat1.cols());
            }
            return output;
        }
        T FindMax(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat1, int m, int n){
            T max = numeric_limits<T>::min();
            for (int i = 0; i < n; i ++){
                if (max < mat1(m, i)) max = mat1(m, i);
            }//end for
            return max;
        }
        int FindMaxIndex(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat1, int m, int n){
            T max = numeric_limits<T>::min();
            int res = 0;
            for (int i = 0; i < n; i ++){
                if (max < mat1(m, i)) {
                    max = mat1(m, i);
                    res = i;
                }
            }//end for
            return res;
        }
        void MySoftmax(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat, int m, int n){
            T sum = 0.0;
            T max = FindMax(mat, m, n);
            for (int i = 0; i < n; i ++){
                T fi = exp(mat(m, i) - max);
                mat(m, i) = fi;
                sum = sum + fi;
            }
            for (int i = 0; i < n; i ++){
                mat(m, i) = mat(m, i) / sum;
            }
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SoftmaxOp(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat1, bool row_or_col){
            /*
                True: Softmax According to Row
                False: Softmax According to Col
            */
            if (!row_or_col){
                return SoftmaxOp(mat1.transpose(), !row_or_col).transpose();
            }
            for (int i = 0; i < mat1.rows(); i ++){
                MySoftmax(mat1, i, mat1.cols());
            }
            return mat1;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
    };

    template <typename T>
    std::ostream& operator<<(std::ostream &os,  Eigen_2D<T> &m){
        os << "[" << m.data << "]";
        return os;
    }

    template <typename T>
    struct Eigen_3D{
        //represent one picture or kernel
    public:
        Eigen_3D(){}
        Eigen_3D(int a, int b, int c, Json::Value node){
            for (int i = 0; i < c ; i ++){
                Eigen_2D<T> temp(a, b);
                for (int m = 0; m < a; m ++){
                    for (int n = 0; n < b; n ++){
                        temp.getData()(m, n) = (T)node[m][n][i].asDouble();
                    }
                }
                this->Tdata.push_back(temp);
            }//end for
        }
        Eigen_3D(int a, int b, int c){
            for (int i = 0; i < c; i ++){
                Eigen_2D<T> temp(a, b);
                this->Tdata.push_back(temp);
            }
        }
        int Tsize(){
            return this->Tdata.size();
        }
        Eigen_2D<T>& operator[](int index){
            return this->Tdata[index];
        }
        template <typename U>
        friend std::ostream& operator<<(std::ostream &os, Eigen_3D<U> &m);

        void initialize(int a, int b, int c){
            for (int i = 0; i < c; i ++){
                Eigen_2D<T> temp(a, b);
                this->Tdata.push_back(temp);
            }
        }
        void setData (std::vector<Eigen_2D<T>>& Tdata){
            this->Tdata.assign(Tdata.begin(), Tdata.end());
        }
        void setData(int a, int b, int c, Json::Value node){
            for (int i = 0; i < c ; i ++){
                Eigen_2D<T> temp(a, b);
                for (int m = 0; m < a; m ++){
                    for (int n = 0; n < b; n ++){
                        temp(m, n) = (T)node[m][n][i].asDouble();
                    }
                }
                this->Tdata.push_back(temp);
            }//end for
        }
        vector<Eigen_2D<T>>& getData(){
            return this->Tdata;
        }
        size_t rows(){
            assert(this->Tdata.size() != 0); //check if we initialized the Tdata
            return this->Tdata[0].rows();
        }
        size_t cols(){
            assert(this->Tdata.size() != 0); //check if we initialized the Tdata
            return this->Tdata[0].cols();
        }
        Eigen_2D<T> convd(Eigen_3D<T> kernel, std::vector<int> stride, padding_type padding){
            assert(this->Tsize() > 0 && kernel.Tsize() > 0 && this->Tsize() == kernel.Tsize());
            Eigen_2D<T> res = this->Tdata[0].convd_base(kernel[0], stride, padding);
            for (int i = 1; i < this->Tdata.size(); i ++){
                res = res.AddWithoutBroadCast(this->Tdata[i].convd_base(kernel[i], stride, padding));
            }
            return res;
        }
    private:
        std::vector<Eigen_2D<T>> Tdata;
    };

    template <typename U>
    std::ostream& operator<<(std::ostream &os, Eigen_3D<U> &m){
        os << "[" << endl;
        for (int i = 0; i < m.Tsize(); i ++){
            cout << m[i];
            if (i != m.Tsize() - 1){
                os << "," << endl << endl;
            }else{
                os << endl;
            }
        }
        os << "]" << endl;
        return os;
    }

    enum struct matrix_type{
        kernel,
        image
    };

    template <typename T>
    struct Eigen_4D{
    public:
        Eigen_4D(){}
        Eigen_4D(int a, int b, int c, int d, Json::Value node, matrix_type mtype){
            this->mtype = mtype;
            if (this->mtype == matrix_type::image){
                for (int i = 0; i < a; i ++){
                    Eigen_3D<T> temp(b, c, d, node[i]);
                    this->Qdata.push_back(temp);
                }//end for
            }else{
                for (int i = 0; i < d; i ++){
                    Eigen_3D<T> temp_3d;
                    for (int m = 0; m < c; m ++){
                        Eigen_2D<T> temp_2d(a,b);
                        for (int n = 0; n < a; n ++){
                            for (int j = 0; j < b; j ++){
                                temp_2d(n, j) = (T)node[n][j][m][i].asDouble();
                            }
                        }
                        temp_3d.getData().push_back(temp_2d);
                    }
                    this->Qdata.push_back(temp_3d);
                }
            }
        }
        Eigen_4D(int a, int b, int c, int d){
            for (int i = 0; i < a; i++){
                Eigen_3D<T> temp(b, c, d);
                this->Qdata.push_back(temp);
            }
        }
        int Qsize(){
            return this->Qdata.size();
        }
        Eigen_3D<T>& operator[] (int index){
            return this->Qdata[index];
        }
        template <typename U>
        friend std::ostream& operator<<(std::ostream &os, const Eigen_4D<U> &m);
        void Initialize(int a, int b, int c, int d){
            for (int i = 0; i < a; i++){
                Eigen_3D<T> temp(b, c, d);
                this->Qdata.push_back(temp);
            }
        }
        void setData(std::vector<Eigen_3D<T>> Qdata){
            this->Qdata.assign(Qdata.begin(), Qdata.end());
        }
        void setData(int a, int b, int c, int d, Json::Value node, matrix_type mtype){
            this->mtype = mtype;
            if (this->mtype == matrix_type::image){
                for (int i = 0; i < a; i ++){
                    Eigen_3D<T> temp(b, c, d, node[i]);
                    this->Qdata.push_back(temp);
                }//end for
            }else{
                for (int i = 0; i < d; i ++){
                    Eigen_3D<T> temp_3d;
                    for (int m = 0; m < c; m ++){
                        Eigen_2D<T> temp_2d(a,b);
                        for (int n = 0; n < a; n ++){
                            for (int j = 0; j < b; j ++){
                                temp_2d(n, j) = (T)node[n][j][m][i].asDouble();
                            }
                        }
                        temp_3d.getData().push_back(temp_2d);
                    }
                    this->Qdata.push_back(temp_3d);
                }
            }//end else
        }
        void getShape(std::vector<int>& res){
            res.push_back(this->Qdata.size());
            res.push_back(this->Qdata[0].rows());
            res.push_back(this->Qdata[0].cols());
            res.push_back(this->Qdata[0].getData().size());
        }
        std::vector<Eigen_3D<T>>& getData(){
            return this->Qdata;
        }
        Eigen_4D<T> convd_with_multi_filter(Eigen_4D<T> kernel, std::vector<int> stride, padding_type padding){
            assert(this->mtype == matrix_type::image); // to ensure the validity of this function
            Eigen_4D<T> res;
            for (int i = 0; i < this->Qdata.size(); i ++){
                Eigen_3D<T> unit_res;
                for (int j = 0; j < kernel.getData().size(); j ++){
                    //number of filters
                    unit_res.getData().push_back(this->Qdata[i].convd(kernel[j], stride, padding));
                }//end for
                res.getData().push_back(unit_res);
            }
            return res;
        }
        Eigen_4D<T> AddBoradCast(Eigen_Vector<T> other){
            Eigen_4D<T> res;
            res.setData(this->Qdata);
            //loop thourght all image and do activation
            for (int i = 0; i < this->Qdata.size(); i ++){
                Eigen_3D<T> &t_3d = res.getData()[i];
                for (int j = 0; j < t_3d.getData().size(); j ++){
                    t_3d[j].AddBoradCast(other, j);
                }
            }
            return res;
        }
        Eigen_4D<T> apply(const std::function<T(const T&)> &f){
            Eigen_4D<T> res;
            res.setData(this->Qdata);
            //Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& f_data = res.getData();
            //loop thourght all image and do activation
            for (int i = 0; i < res.Qsize(); i ++){
                for (int j = 0; j < res[i].Tsize(); j ++){
                    res[i][j] = res[i][j].apply(f);
                }
            }
            return res;
        }
        /*
            We do not do reshape on kernel matrix, normally we reshape placeholder matrix [a,b,c,d]
            -a : number of pictures
            -b : dimension x
            -c : dimension y
            -d : number of channelsT
            Note that we reshape a 4-dimension to prepare for the future fully connected layer connection
        */
        #define CALCULATE_INDEX(channel_index, num_of_c, col_size, i, j) (channel_index + col_size*num_of_c*i + num_of_c*j)
        Eigen_2D<T> reshape(){
            assert(this->Qdata.size() > 0); //To ensure the correctness of the following code
            Eigen_2D<T> res(this->Qdata.size(), this->Qdata[0].rows()*this->Qdata[0].cols()*this->Qdata[0].getData().size());
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& res_data = res.getData();
            size_t num_of_c = this->Qdata[0].Tsize();
            for (int m = 0; m < this->Qsize(); m ++){
                // Efficient loop through 3D map
                for (int n_c = 0; n_c < num_of_c; n_c ++){
                    //fetch one specific channel
                    Eigen_2D<T> image_map = this->Qdata[m][n_c];
                    for (int i = 0; i < image_map.rows(); i++){
                        for (int j = 0; j < image_map.cols(); j ++){
                            res_data(m, CALCULATE_INDEX(n_c, num_of_c,image_map.cols(), i, j)) = image_map(i,j);
                        }
                    }//loop through the image
                }
            }
            return res;
        }
    private:
        std::vector<Eigen_3D<T>> Qdata;
        matrix_type mtype = matrix_type::image;
    };
    template <typename T>
    std::ostream& operator<<(std::ostream &os, Eigen_4D<T> &m){
        os << "[" << endl;
        for (int i = 0; i < m.Qsize(); i ++){
            cout << m[i];
            if (i != m.Qsize() - 1){
                os << "," << endl;
            }else{
                os << endl;
            }
        }
        os << "]" << endl;
        return os;
    }
}

#endif //C_INFERENCE_DATATYPE_H
