#include <torch/extension.h>
#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

#define AT_DISPATCH_HAS_VALUE(optional_value, ...) \
    [&] {                                          \
        if (optional_value.has_value())            \
        {                                          \
            const bool HAS_VALUE = true;           \
            return __VA_ARGS__();                  \
        }                                          \
        else                                       \
        {                                          \
            const bool HAS_VALUE = false;          \
            return __VA_ARGS__();                  \
        }                                          \
    }()

torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace);

// Returns `rowptr`, `col`, `n_id`, `e_id`
//这里的函数应该是采样，那么返回的应该是点和边，那么返回rowptr以及col是为什么呢？
torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace)
{
    //检查数据是否存在于CPU上
    CHECK_CPU(rowptr);
    CHECK_CPU(col);
    CHECK_CPU(idx); //是nodes吗?
    CHECK_INPUT(idx.dim() == 1);
    //auto自动变量声明符,类似于C语言中的局部变量,使用完后就不再存在了
    //定义了三个tensor的指针,data_ptr是指针,data_ptr<int62_t>是利用模板方法指定指针类型
    auto rowptr_data = rowptr.data_ptr<int64_t>(); //每行的索引
    auto col_data = col.data_ptr<int64_t>();  //数据
    auto idx_data = idx.data_ptr<int64_t>();  //每个数据的列索引
    //vector是可以改变大小的数组的序列容器
    std::vector<int64_t> n_ids;

    int64_t i;
    

    int64_t n, c, e, row_start, row_end, row_count;
    //不需要进行采样的情况
    if (num_neighbors < 0)
    { // No sampling ======================================
        //两层循环
        //对每一个节点,对每一行进行一个观察，对每一个节点的邻居进行采样
        for (int64_t i = 0; i < idx.numel(); i++)
        {
            //获取每一行数据的个数
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start; //当前行存储的元素个数
            //所以这个按行压缩存储的到底是个什么内容
            for (int64_t j = 0; j < row_count; j++)
            {
                e = row_start + j;
                c = col_data[e];  //实际压进去的是图节点数据的特征
                n_ids.push_back(c); //将所有的数据全部压进去
            }
        }
    }
    //这个replace是干嘛用的
    else if (replace)
    { // Sample with replacement ===============================
        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];    //n对应的应该是行索引,则idx_data对应的是数据的行索引
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            std::unordered_set<int64_t> perm;
            //重复采样
            //若总的邻居数目小于需要采样的邻居数目，则在已有邻居中再随机采样剩余的次数
            if (row_count <= num_neighbors)
            {
                for (int64_t j = 0; j < row_count; j++)
                    perm.insert(j);
                for (int64_t j = 0; j < num_neighbors-row_count; j++){
                    e = row_start + rand() % row_count;
                    c = col_data[e];
                    n_ids.push_back(c);
                }
            }
            else
            { // See: https://www.nowherenearithaca.com/2013/05/
                //      robert-floyds-tiny-and-beautiful.html
                for (int64_t j = row_count - num_neighbors; j < row_count; j++)
                {
                    //随机采样,所有这个压缩矩阵到底是怎么存储的呢
                    if (!perm.insert(rand() % j).second)
                        perm.insert(j);
                }
            }

            
            for (const int64_t &p : perm)
            {
                e = row_start + p;
                c = col_data[e];
                n_ids.push_back(c);
            }
            
        }
        // for (int64_t i = 0; i < idx.numel(); i++)
        // {
        //     n = idx_data[i];
        //     row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
        //     row_count = row_end - row_start;
        //     // std::vector<int64_t>().swap(temp);
        //     // for (int64_t j = 0; j < row_count; j++)
        //     // {
        //     //     temp.push_back(j);
        //     // }
        //     // if (row_count<num_neighbors){
        //     //     for (int64_t j = 0; j <num_neighbors-row_count; j++){
        //     //         temp.push_back(rand() % row_count);
        //     //     }
        //     // }
        //     // std::random_shuffle(temp.begin(), temp.end());
        //     std::unordered_set<int64_t> perm;
        //     for (int64_t j = 0; j < num_neighbors; j++)
        //     {
        //         e = row_start + rand() % row_count;
        //         // e = row_start + temp[j];
        //         c = col_data[e];
        //         n_ids.push_back(c);
        //     }
        // }
    }
    else
    { // Sample without replacement via Robert Floyd algorithm ============

        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            std::unordered_set<int64_t> perm;
            if (row_count <= num_neighbors)
            {
                for (int64_t j = 0; j < row_count; j++)
                    perm.insert(j);
            }
            else
            { // See: https://www.nowherenearithaca.com/2013/05/
                //      robert-floyds-tiny-and-beautiful.html
                for (int64_t j = row_count - num_neighbors; j < row_count; j++)
                {
                    if (!perm.insert(rand() % j).second)
                        perm.insert(j);
                }
            }

            for (const int64_t &p : perm)
            {
                e = row_start + p;
                c = col_data[e];
                n_ids.push_back(c);
            }
        }
    }

    int64_t N = n_ids.size();
    auto out_n_id = torch::from_blob(n_ids.data(), {N}, col.options()).clone();

    return out_n_id;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sample_neighber_cpu", &sample_neighber_cpu, "Node neighborhood sampler");
}