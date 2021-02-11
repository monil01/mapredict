<!-- # MAPredict -->

![](model_parser/Mapredict.png)

#### A static analysis driven memory access prediction framework for modern CPUs.



```diff
++ New: MAPredict now does the total nodal prediction of LULESH. > 90% accuracy in all intel micro-architecutres. Source code: [applications/memory_research_ornl/aspen_model_generation/lulesh_full]++ 
```

#### 1. Overview
##### MAPredict, a static analysis driven framework that provides memory access prediction by gathering application and machines properties at compile time. MAPredict invokes an analytical model to predict LLC-DRAM traffic by combining the application model, the machine model, and user-provided hints for capturing dynamic information. MAPredict is tested on different micro-architectures of Intel and provided high accuracy for applications with different access patterns.

#### 2. Organization of the Repository
    2.1 [applications]: Follow the installation procedure to get all the content of this folder.  
          2.1.1 [applications/memory_research_ornl/all_apps_experiments]:  TAU-PAPI script to generate LLC-DRAM traffic data for applications. scripts are also available for different micro-architechtures of Intel and also for OpenMP execution.
          2.1.2 [applications/memory_research_ornl/aspen_model_generation]: This folder contains all model generation source code (annotated source code) which is used by MAPredict to generate application model.
    2.2 [aspen]: this folder contains aspen code which is needed for MAPredict. Aspen abstract models enable the creation of performance models.
    2.3 [OpenARC-devel]: This folder contains jar files (compiled) of the OpenARC and COMPASS framework. MAPredict is built on top of COMPASS and OpenARC. However, it is not open source and for this reason, only the java binaries are provided.
    2.4 [model_parser]: This folder contains MAPredict's model parser that traverses the application and machine model and invokes an appropriate analytical model.
    2.5 [models]: This folder contains the application and machine model which are passes to MAPredict for prediction.
    2.6 [scripts]: These scripts are used to compile MAPredict and invoke appropriate machine and application models.
    

#### 3. Installation procedure.

    3.1 Prerequisite
        3.1.1 GCC 5.4 or higher.
        3.1.2 Java
        3.1.3 Bison
        3.1.4 python 3.6 or higher.
        3.1.5 cuda-10 or higher
        
    3.2 Getting the code
        3.2.1 git clone https://github.com/monil01/mapmc.git
        3.2.2 cd mapmc
        3.2.3 git checkout master
        3.2.4 git submodule init
        3.2.5 git submodule update
        
    3.3 installing
        3.3.1 cd [MAPredict_root]/aspen
        3.3.2 ./configure
        3.3.3 make all -j (it should build libaspen.a in the lib folder)
        3.3.4 cd .. (come out of aspen directory)
        3.3.5 cd OpenARC-devel
        3.3.5 export openarc=`pwd`
        3.3.6 make (it will create all openarc binaries and drivers)
        3.3.7 cd ..
        3.3.8 Modify the python path in the Makefile.
        3.3.9 make (it should build the MAPredict binary)
        
    3.4 testing
        3.4.1 run: ./scripts/stream_100.sh  , it should compile mapredict, generate the application model and generate a prediction result.
        3.4.2 convention of running MAPredict: ./[MAPredict_binary] [application_model] [machine_model]
  
  
#### 4. Examples

    4.1 Source Code annonation: Source code annotation examplese are in [applications/memory_research_ornl/aspen_model_generation].
    4.2 Memory prediction for all applications: [./scripts] directory has scripts for all applications that generate memory prediction.
    4.3 #### Lulesh : source code annotation processor for lulesh can be found in [applications/memory_research_ornl/aspen_model_generation].
    

#### 5. Data for ICS21
    5.1 ICS21 submission data is available at [applications/memory_research_ornl/data_paper/Data_for_ICS21.xlsx].


  
    
    
  
