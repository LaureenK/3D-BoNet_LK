# py2.7 + TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py27tf12/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py2.7 + TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/klein/localcuda/myVenvTest/local/lib/python2.7/site-packages/tensorflow/include -I /home/klein/localcuda/cuda-8.0/include -I /home/klein/localcuda/myVenvTest/local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /home/klein/localcuda/cuda-8.0/lib64/ -L/home/klein/localcuda/myVenvTest/local/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py36tf12/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py36tf14/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /home/bo/miniconda2/envs/py36tf14/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/bo/miniconda2/envs/py36tf14/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
