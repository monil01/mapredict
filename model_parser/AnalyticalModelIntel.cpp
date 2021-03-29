// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/RuntimeCounter.h"
#include "walkers/RuntimeExpression.h"

#include "model/ASTDataStatement.h"

#include "app/ASTExecutionBlock.h"
#include "model/ASTMachModel.h"
#include "app/ASTRequiresStatement.h"
#include "walkers/AspenTool.h"

#include "Types.h"
#include "AnalyticalModelIntel.h"

#define Finegrained_RSC_Print

//using namespace std;
//

/*
AnalyticalModelIntel(
    int compiler,
    int instruction_type,
    size_t cache_line_size,
    vector<ASTTrait*> traits,
    bool prefetch_enabled,
    bool multithreaded,
    int64_t data_structure_size,
    int element_size
);

*/


AnalyticalModelIntel::AnalyticalModelIntel(
    int compiler,
    int instruction_type,
    int cache_line_size,
    vector<ASTTrait*> traits,
    bool prefetch_enabled,
    bool multithreaded,
    std::int64_t data_structure_size,
    int element_size,
    int microarchitecture 
)   
     
{ 
    _compiler = compiler;
    _instruction_type = instruction_type;
    _cache_line_size = cache_line_size;
    _traits = traits;
    _prefetch_enabled = prefetch_enabled;
    _multithreaded = multithreaded;
    _data_structure_size = data_structure_size;
    _element_size = element_size;
    _microarchitecture = microarchitecture;
    _analytical_model_utility_obj = new AnalyticalModelUtility();


    if(DEBUG_MAPMC == true) std::cout << " Analytical model trait size " <<  traits.size() << std::endl;

    _access_pattern = _analytical_model_utility_obj->findPattern(_traits);
    _initialized = _analytical_model_utility_obj->findInitialized(_traits);
    if(DEBUG_MAPMC == true) std::cout << " Analytical model access pattern " <<  _access_pattern << " init " << _initialized << std::endl;

    _pattern_string = _analytical_model_utility_obj->findPatternString(_traits);
    if(DEBUG_MAPMC == true) std::cout << " Analytical model pattern string " <<  _pattern_string << std::endl;



    if ( _element_size * _data_structure_size > 4 * KB ) _page_size = 2 * MB;

    //predictMemoryAccess();
        


}

AnalyticalModelIntel::~AnalyticalModelIntel(){

if(_analytical_model_utility_obj)   
    delete _analytical_model_utility_obj;
}



std::int64_t AnalyticalModelIntel::predictMemoryAccess(){

    std::int64_t memory_access = 0;
    
    //if(DEBUG_MAPMC == true) std::cout << " Calling Stencil prediction " <<  _access_pattern << " init " << _initialized << access_patterns::STREAM << access_patterns::STRIDE << access_patterns::STENCIL << access_patterns::RANDOM << std::endl;
    if ( _access_pattern == access_patterns::STREAM) {
        if(DEBUG_MAPMC == true) std::cout << " Calling Stream prediction " <<  _access_pattern << " init " << _initialized << std::endl;
        memory_access = streamAccess();
    } else if ( _access_pattern == access_patterns::STRIDE) {
        if(DEBUG_MAPMC == true) std::cout << " Calling Stride prediction " <<  _access_pattern << " init " << _initialized << std::endl;
        memory_access = strideAccess();
    } else if ( _access_pattern == access_patterns::STENCIL) {
        if(DEBUG_MAPMC == true) std::cout << " Calling Stencil prediction " <<  _access_pattern << " init " << _initialized << std::endl;
        memory_access = stencilAccess();
    } else if ( _access_pattern == access_patterns::RANDOM)  {
        if(DEBUG_MAPMC == true) std::cout << " Calling Random prediction " <<  _access_pattern << " init " << _initialized << std::endl;
        memory_access = randomAccess();
    }

    return memory_access;
}

std::int64_t  AnalyticalModelIntel::streamAccess() {
    std::int64_t memory_access = 0;
    /// renaming the variables as per model described in the paper
    std::int64_t N = _data_structure_size;
    int CL = _cache_line_size;
    int ES = _element_size;
    // converting the stride and data size to bytes
    
    if (_compiler == compilers::GCC) {
        if (_initialized == true) {
            if ( _instruction_type == instructions::LOAD) {
                //memory_access =  N * ES / (double) CL/10000 ;
                //memory_access = ceil( N * ES / (double) CL/10000 );
                memory_access = ceil( N * ES / (double) CL ) * CL;
                if(DEBUG_MAPMC == true) std::cout << " STREAM Memory access region 1 - 1  : " << memory_access << std::endl;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = 2 * ceil( N * ES / (double) CL ) * CL;
                if(DEBUG_MAPMC == true) std::cout << " STREAM Memory access region 1 - 2  : " << memory_access << std::endl;
            }
        } else {
            if ( _instruction_type == instructions::LOAD) {
                memory_access = ceil( N * ES / (double) CL ) * CL;
                if(DEBUG_MAPMC == true) std::cout << " STREAM Memory access region 1 - 3  : " << memory_access << std::endl;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = ceil( N * ES / (double) _page_size ) * _page_size;
                if(DEBUG_MAPMC == true) std::cout << " STREAM Memory access region 1 - 4  : " << memory_access << std::endl;
            }
        }
    } 
    
    memory_access = ceil( memory_access / (double) CL );

    if(DEBUG_MAPMC == true) std::cout << " Analytical Model STREAM " << memory_access << " data size "  
        <<  N  << " element size " << ES << " cacheline " 
        << CL << " page size " << _page_size  << "  instruction " << _instruction_type << std::endl;

    return memory_access;

}


std::int64_t AnalyticalModelIntel::randomAccess(){

    std::int64_t memory_access = 0;
    double access = 0;
    std::string algorithm = _analytical_model_utility_obj->findAlgorithm(_traits);
    int factor = _analytical_model_utility_obj->findFactor(_traits); // Using stride we transer any factor that can be used.
    std::int64_t N = _data_structure_size;
    std::string correction_str = _analytical_model_utility_obj->generateCorrectionString(_microarchitecture, _prefetch_enabled);
    double correction = _analytical_model_utility_obj->findCorrection(_traits, correction_str);

    if (algorithm == "logarithm"){
        //access = log(N)/ log(2); 
        access = log10(N); 
        if(DEBUG_MAPMC == true) std::cout << " Random logarithm " << access << std::endl;
    }
    
    memory_access = access * factor;
    if(correction != 1) memory_access = memory_access * correction;
 
    if(DEBUG_MAPMC == true) std::cout << " Analytical Model RANDOM " << memory_access << " data size "  
        <<  N   << " algorithm " << algorithm  << "  instruction " << _instruction_type << std::endl;
    /************* TODO ***************
 *
 * there could be three kinds of randomness
 * 1   algorithmic randomness
 * 2    Cache thrashing
 * 3    Data structure access randomness
 *
 * We currently have provision for algorithmic randomness. 
 * other algorithmic randomness needs to cover.
 * Here it is for the binary search
 * ***********************************/

    //return memory_access; 
    /* 
    if (_compiler == compilers::GCC) {
        if (_initialized == true) {
            if ( _instruction_type == instructions::LOAD) {
                memory_access = ceil( N * ES / (double) CL ) * CL;
                if(DEBUG_MAPMC == true) std::cout << " Memory access " << memory_access << std::endl;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = 2 * ceil( N * ES / (double) CL ) * CL;
            }
        } else {
            if ( _instruction_type == instructions::LOAD) {
                memory_access = ceil( N * ES / (double) CL ) * CL;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = ceil( N * ES / (double) _page_size ) * _page_size;
            }
        }
    } 
    */
 
   

    return memory_access;
}

std::int64_t AnalyticalModelIntel::strideAccess(){
    int S = _analytical_model_utility_obj->findStride(_traits);
    std::int64_t memory_access = 0;
    /// renaming the variables as per model described in the paper
    std::int64_t N = _data_structure_size * S;
     _data_structure_size = N;
    if(DEBUG_MAPMC == true) std::cout << " Stride data structure size " << N << std::endl;
    int CL = _cache_line_size;
    int ES = _element_size;
    int PS = _page_size;
    // converting the stride and data size to bytes

    // Streaming zone
    if( S*ES <= CL) {
        //_data_structure_size = N * S;
        memory_access = streamAccess() * CL;
        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory acces region 6 - 1 " << memory_access << std::endl; 
    }
    // No prefetching zone
    else if( S*ES >  5* CL) {
        //_data_structure_size = N * S;
            if (_initialized == true) {
                memory_access = streamAccess() * CL;
                memory_access = memory_access * CL / (S * ES);
                if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 8 - 1  : " << memory_access << std::endl;
            } else {
                if (S <= PS){
                    if (_instruction_type == instructions::STORE) {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * 1; //same as streaming access
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 8 - 2  : " << memory_access << std::endl;
                    }
                    else if(_instruction_type == instructions::LOAD)  {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * CL / (S * ES);
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 8 - 3  : " << memory_access << std::endl;
                    }
                } else { 
                    if (_instruction_type == instructions::STORE)  {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * PS/ (S * ES); //same as streaming access
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 8 - 4  : " << memory_access << std::endl;
                    }
                    else if(_instruction_type == instructions::LOAD)  {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * CL / (S * ES);
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 8 - 5  : " << memory_access << std::endl;
                    }
                }
            }

        //if(DEBUG_MAPMC == true) std::cout << " Memory access region 8 " << memory_access << std::endl; 
    } 
    // prefetching zone
    else {
        //_data_structure_size = N * S;

        // prefetching disabled 
        if (_prefetch_enabled == 0) {
            if (_initialized == true) {
                memory_access = streamAccess() * CL;
                memory_access = memory_access * CL / (S * ES);
                if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 1  : " << memory_access << std::endl;
            } else {
                if (S <= PS){
                    if (_instruction_type == instructions::STORE) {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * 1; //same as streaming access
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 2  : " << memory_access << std::endl;
                    }
                    else if(_instruction_type == instructions::LOAD)  {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * CL / (S * ES);
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 3  : " << memory_access << std::endl;
                    }
                } else { 
                    if (_instruction_type == instructions::STORE)  {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * 1; //same as streaming access
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 4  : " << memory_access << std::endl;
                    }
                    else if(_instruction_type == instructions::LOAD)  {
                        memory_access = streamAccess() * CL;
                        memory_access = memory_access * PS/ (S * ES); 
                        if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 5  : " << memory_access << std::endl;
                    }
                }
            }

        // prefetching enabled 
        } else {
            if (_initialized == true) {
                if ( _instruction_type == instructions::LOAD) {
                    memory_access = 3 * CL * (N/(double)S);
                    //check microarchitecture
                    //if (_microarchitecture == microarchitecture::BW) double factor = 1;
                    if (_microarchitecture == microarchitecture::SK || _microarchitecture == microarchitecture::CS || _microarchitecture == microarchitecture::CP ) {
                        double factor = 0.85;
                        memory_access = (std::int64_t) (memory_access * factor);
                    }

                    if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 6  : " << memory_access << std::endl;
                } else if ( _instruction_type == instructions::STORE) {
                    memory_access = streamAccess() * CL;
                    memory_access = memory_access * CL / (S * ES);
                    if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 7  : " << memory_access << std::endl;
                }
            } else { 
                if ( _instruction_type == instructions::LOAD) {
                    memory_access = streamAccess() * CL;
                    memory_access = 3 * CL * (N/(double)S);
                    if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 8  : " << memory_access << std::endl;
                } else if ( _instruction_type == instructions::STORE) {
                    memory_access = streamAccess() * CL;
                    memory_access = memory_access * 1; //same as streaming access
                    if(DEBUG_MAPMC == true) std::cout << " STRIDE Memory access region 7 - 9  : " << memory_access << std::endl;
                }
            }
        }
        //if(DEBUG_MAPMC == true) std::cout << " Memory access region 8" << memory_access << std::endl; 
    }

    memory_access = ceil( memory_access / (double) CL );
    
    if(DEBUG_MAPMC == true) std::cout << " Analytical Model STRIDE " << memory_access << " data size "  
        <<  N  << " element size " << ES << " cacheline "
        << CL << " page size " << _page_size  << "  instruction " << _instruction_type << std::endl;

    return memory_access;

}
std::int64_t AnalyticalModelIntel::stencilAccess(){
    std::int64_t memory_access = 0;
    /// renaming the variables as per model described in the paper
    std::int64_t N = _data_structure_size;
    int CL = _cache_line_size;
    int ES = _element_size;
    // converting the stride and data size to bytes

    if (_pattern_string == "stencil4"){

        _data_structure_size = _data_structure_size / 4;

        if(DEBUG_MAPMC == true) std::cout << " Stencil data size reduction  : " << _pattern_string << " data size " << _data_structure_size << std::endl;

    }
    if (_pattern_string == "stencil5"){
        _data_structure_size = _data_structure_size / 5;
        if(DEBUG_MAPMC == true) std::cout << " Stencil data size reduction  : " << _pattern_string << " data size " << _data_structure_size << std::endl;
    }
    if (_pattern_string == "stencil8"){
        _data_structure_size = _data_structure_size / 8;
        if(DEBUG_MAPMC == true) std::cout << " Stencil data size reduction  : " << _pattern_string << " data size " << _data_structure_size << std::endl;
    }
    if (_pattern_string == "stencil27"){
        _data_structure_size = _data_structure_size / 27;
        if(DEBUG_MAPMC == true) std::cout << " Stencil data size reduction  : " << _pattern_string << " data size " << _data_structure_size << std::endl;
    }
  
    memory_access = streamAccess() * CL;

    std::string reuse_str = _analytical_model_utility_obj->generateReuseString(_microarchitecture, _prefetch_enabled);
    double reuse_factor = _analytical_model_utility_obj->findReuseFactor(_traits, reuse_str);


    //return memory_access; 
    /* 
    if (_compiler == compilers::GCC) {
        if (_initialized == true) {
            if ( _instruction_type == instructions::LOAD) {
                memory_access = ceil( N * ES / (double) CL ) * CL;
                if(DEBUG_MAPMC == true) std::cout << " Memory access " << memory_access << std::endl;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = 2 * ceil( N * ES / (double) CL ) * CL;
            }
        } else {
            if ( _instruction_type == instructions::LOAD) {
                memory_access = ceil( N * ES / (double) CL ) * CL;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = ceil( N * ES / (double) _page_size ) * _page_size;
            }
        }
    } 
    */
    
    if(reuse_factor > 1) memory_access = memory_access * reuse_factor;

    memory_access = ceil( memory_access / (double) CL );
    if(DEBUG_MAPMC == true) std::cout << " Analytical Model STENCIL " << memory_access << " data size "  
        <<  N  << " element size " << ES << " cacheline " 
        << CL << " page size " << _page_size  << "  instruction " << _instruction_type << std::endl;
    
    return memory_access;

}



