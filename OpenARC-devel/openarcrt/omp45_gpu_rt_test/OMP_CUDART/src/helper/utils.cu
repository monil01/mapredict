#include "utils.h"

void print2DData(FILE* output, double* data, long width, long height)
{
	long k=0;
	for (int i=0; i< height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fprintf(output, " %2.1f\t",data[k++]);
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}

void print2DData(FILE* output, float* data, long width, long height)
{
	long k=0;
	for (int i=0; i< height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fprintf(output, " %2.1f\t",data[k++]);
		}
		fprintf(output, "\n");
	}
	fprintf(output, "\n");
}


TIME_MSEC get_current_msec()
{
	struct timeval time;
	gettimeofday (&time, NULL);

	TIME_MSEC msec;
	msec = (time.tv_sec*1000000);
	msec += (time.tv_usec);
	return msec;
}

char* getArgumentValue(int argc, char **argv, char *argName)
{
	char* result = NULL;

	for (int i=0; i<argc; i++)
	{
		char* substr = strstr(argv[i], argName);
		if (substr!= NULL)
		{
			result = strtok (substr,"=");
			if (result != NULL)
			{
				result = strtok (NULL, "=");
			}
			break;
		}
	}

	return result;
}

int msleep(unsigned long milisec)
{
	struct timespec req={0};
	time_t sec=(int)(milisec/1000);
	milisec=milisec-(sec*1000);
	req.tv_sec=sec;
	req.tv_nsec=milisec*1000000L;
	while(nanosleep(&req,&req)==-1)
		continue;
	return 1;
}

long safeSize(long size)
{
	if (size < SAFE_SIZE)
		size = SAFE_SIZE;

	return size;
}

void setIntValue(int* lvalue, char* rvalue, int defaultValue){
	if (rvalue != NULL){
		*lvalue = atoi(rvalue);
	}
//	else{
//		*lvalue = defaultValue;
//	}
}


void setFloatValue(float* lvalue, char* rvalue, float defaultValue){
	if (rvalue != NULL){
		*lvalue = atof(rvalue);
	}
//	else{
//		*lvalue = defaultValue;
//	}
}

void parseParams(int argc, char *argv[]){
	char* paramsFileName = NULL;

	// This is the case no parameters given. In this case it will read values
	// default params file
	if (argc > 1) {
		char tempstr[] = "pf";
		paramsFileName=getArgumentValue(argc,argv,tempstr);
	}

	if (paramsFileName != NULL) {
		ifstream file(paramsFileName);
	    string str;
	    argv = new char*[20]();
		char tempstr[] = "";

	    argv[0]= tempstr;
	    argc=1;

	    while (std::getline(file, str))
	    {
	    	const char * lineArray = str.c_str();
	    	if (strlen(lineArray)==0)
	    		continue;

	    	char firstChar = lineArray[0];
	    	if (firstChar != '\t' && firstChar != '#'){
				argv[argc] = new char[30]();
				strcpy(argv[argc],str.c_str());
				argc++;
	    	}
	    }
	    file.close();
	}
}

