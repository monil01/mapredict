#include <jni.h>
#include <iostream>
using namespace std;

// JNI includes
#include "aspen/AppModel.h"
#include "aspen/Expression.h"
#include "aspen/MachModel.h"
#include "aspen/MachComponent.h"
#include "aspen/Aspen.h"

// Aspen includes
#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "mach/ASTMachComponent.h"
#include "expr/Expression.h"
#include "parser/Parser.h"

// ----------------------------------------------------------------------------
// AppModel
// ----------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_aspen_AppModel_nativeFinalize(JNIEnv *env, jobject thisObj, jlong obj)
{
    if (obj)
        delete (ASTAppModel*)(obj);
}

JNIEXPORT jstring JNICALL
Java_aspen_AppModel_nativeGetName(JNIEnv *env, jobject thisObj, jlong obj)
{
    jstring rv = env->NewStringUTF(((ASTAppModel*)obj)->GetName().c_str());
    return rv;
}

JNIEXPORT long JNICALL
Java_aspen_AppModel_nativeGetResourceRequirementExpression(JNIEnv *env, jobject thisObj, jlong obj, jstring str)
{
    const char *res = env->GetStringUTFChars(str, 0);
    jlong rv = (jlong)(((ASTAppModel*)obj)->GetResourceRequirementExpression(res));
    env->ReleaseStringUTFChars(str, res);
    return rv;
}

// ----------------------------------------------------------------------------
// MachModel
// ----------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_aspen_MachModel_nativeFinalize(JNIEnv *env, jobject thisObj, jlong obj)
{
    if (obj)
        delete (ASTMachModel*)(obj);
}

JNIEXPORT jlong JNICALL
Java_aspen_MachModel_nativeGetMachine(JNIEnv *env, jobject thisObj, jlong obj)
{
    return (jlong)(((ASTMachModel*)obj)->GetMachine());
}

// ----------------------------------------------------------------------------
// MachComponent
// ----------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_aspen_MachComponent_nativeFinalize(JNIEnv *env, jobject thisObj, jlong obj)
{
    if (obj)
        delete (ASTMachComponent*)(obj);
}

JNIEXPORT jstring JNICALL
Java_aspen_MachComponent_nativeGetName(JNIEnv *env, jobject thisObj, jlong obj)
{
    jstring rv = env->NewStringUTF(((ASTMachComponent*)obj)->GetName().c_str());
    return rv;
}

// ----------------------------------------------------------------------------
// Expression
// ----------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_aspen_Expression_nativeFinalize(JNIEnv *env, jobject thisObj, jlong obj)
{
    if (obj)
        delete (Expression*)(obj);
}

JNIEXPORT jstring JNICALL
Java_aspen_Expression_nativeGetText(JNIEnv *env, jobject thisObj, jlong obj)
{
    jstring rv = env->NewStringUTF(((Expression*)obj)->GetText().c_str());
    return rv;
}

// ----------------------------------------------------------------------------
// global parser methods
// ----------------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_aspen_Aspen_nativeLoadAppModel(JNIEnv *env, jclass thisClass, jstring str)
{
    const char *fn = env->GetStringUTFChars(str, 0);
    jlong rv = (jlong)(LoadAppModel(fn));
    env->ReleaseStringUTFChars(str, fn);
    return rv;
}

JNIEXPORT jlong JNICALL
Java_aspen_Aspen_nativeLoadMachineModel(JNIEnv *env, jclass thisClass, jstring str)
{
    const char *fn = env->GetStringUTFChars(str, 0);
    jlong rv = (jlong)(LoadMachineModel(fn));
    env->ReleaseStringUTFChars(str, fn);
    return rv;
}
