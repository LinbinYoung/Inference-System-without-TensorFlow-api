#ifndef MULTIDATA_H
#define MULTIDATA_H
#include <common/commonlib.h>
#include <DataType/datatype.h>

namespace MultiEigen{
    enum struct tyname{
        D_4,
        D_3,
        D_2,
        D_1,
        D_0
    };
    template <typename T>
    struct TensorData{
        public:
            TensorData(){}
            TensorData(tyname name){
                this->tname = name;
            }
            tyname& getType(){
                return this->tname;
            }
            void setType(tyname name){
                this->tname = name;
            }
            Eigen_2D<T> E2D;
            Eigen_3D<T> E3D;
            Eigen_4D<T> E4D;
            Eigen_Vector<T> E1D;
            TensorData& operator=(const TensorData<T> & other){
                this->E1D = other.E1D;
                this->E2D = other.E2D;
                this->E3D = other.E3D;
                this->E4D = other.E4D;
                this->tname = other.tname;
                return *this;
            }
        private:
            tyname tname;
    };
}

#endif