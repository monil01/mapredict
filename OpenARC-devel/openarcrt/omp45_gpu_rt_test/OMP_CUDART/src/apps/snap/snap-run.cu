#include "../../rt.h"
#include "../main.h"
#include "snap.h"
#include "../timer.h"


#ifdef APP_SNAP
double run(struct user_parameters* params)
{
	if (params->mode == MODE_GLOBAL){
		return run_snap_global(params);
	}
	else if (params->mode == MODE_TASK){
		return run_snap_task(params);
	}
}
#endif
