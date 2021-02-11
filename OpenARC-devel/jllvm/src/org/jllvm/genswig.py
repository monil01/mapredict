#!/usr/bin/python
import os
import re
import subprocess
import sys

def header_include(name,path,std_include_dir):
	if std_include_dir:
		return '<' + os.path.join(path[len(std_include_dir):],name + '.h') + '>'
	else:
		return '"' + path + '/' + name + '.h"'
		
def runprog(prog,args):
	popen_args = prog
	for arg in args:
		popen_args = popen_args + ' ' + arg
	print popen_args
	proc = subprocess.Popen(popen_args,stdout=subprocess.PIPE,shell=True)
	while proc.poll() == None:
		pass
	print proc.stdout.read()
	
def join_paths(paths):
	def do_joins(path,paths):
		if len(paths) == 1:
			return os.path.join(path,paths[0])
		else:
			return do_joins(os.path.join(path,paths[0]),paths[1:])
	return do_joins(paths[0],paths[1:])
	
def last_token(string,delimiter):
	s = string.split(delimiter)
	if s[len(s)-1] != '':
		return s[len(s)-1]
	else:
		return s[len(s)-2]

def generate_swig_interfaces(llvm_dir):
	assert(os.path.isdir(llvm_dir))
	if llvm_dir[len(llvm_dir)-1] != '/':
		llvm_dir = llvm_dir + '/'
	llvm_include_dir = llvm_dir+ 'include/';
	llvm_c_dir = llvm_include_dir + 'llvm-c/';
	assert(os.path.isdir(llvm_c_dir))
	headers = []
	for root, dirs, files in os.walk(llvm_c_dir):
		for file in files:
			if file.rpartition('.')[2] == 'h':
				headers.append(os.path.join(root, file))
	print 'Generating JNI interface for headers in ' + llvm_c_dir + ' using SWIG.'
	for header in headers:
		name = os.path.basename(header).rpartition('.')[0]

		# We place *.i files for llvm-c/Transforms subdirectory in
		# the llvm-c directory so that they're part of the same Java
		# package. This avoids visibility issues, and it avoids
		# duplicate SWIG_p_* types bewteen directories.
		relpath = os.path.normpath('./' + last_token(llvm_c_dir,'/')) # + '/' + os.path.dirname(header)[len(llvm_c_dir):] + '/')

		print 'Preprocessing ' + header + ' for SWIG at ' + relpath
		if os.path.isdir(relpath) == False:
				os.makedirs(relpath)
		hfile = open(header)
		contents = hfile.read()
		hfile.close()
		out = open(join_paths([relpath,name + '.i']),"w")
		out.write('%module ' + name + '\n%{\n')

		# LLVM doesn't always include all required header files, so
		# the user (SWIG in this case) has to.
		if contents.find('bool') >= 0:
			out.write('#include <stdbool.h>\n')
		if name == 'TargetMachine':
			out.write('#include <llvm-c/Target.h>\n')

                out.write('#include ' + header_include(name,os.path.dirname(header),llvm_include_dir) + '\n')
		out.write('%}\n')

		if contents.find('LLVMTypeRef *') >=0 or contents.find('LLVMValueRef *') >= 0 or contents.find('LLVMGenericValueRef *') >= 0 or contents.find('unsigned *') >=0 or contents.find('LLVMModuleRef *') >= 0 or contents.find('LLVMBasicBlockRef *') >= 0 or contents.find('LLVMExecutionEngineRef *') >= 0 or contents.find('char **') >= 0:
			out.write('\n%include "carrays.i"\n')
		if contents.find('LLVMTypeRef *') >= 0:
			out.write('%array_functions(LLVMTypeRef,LLVMTypeRefArray)\n')
		if contents.find('LLVMValueRef *') >= 0:
			out.write('%array_functions(LLVMValueRef,LLVMValueRefArray)\n')
		if contents.find('LLVMGenericValueRef *') >= 0:
			out.write('%array_functions(LLVMGenericValueRef,LLVMGenericValueRefArray)\n')
		if contents.find('unsigned *') >= 0:
			out.write('%array_functions(unsigned,UnsignedIntArray)\n')
		if contents.find('LLVMModuleRef *') >= 0:
			out.write('%array_functions(LLVMModuleRef,LLVMModuleRefArray)\n')
		if contents.find('LLVMBasicBlockRef *') >= 0:
			out.write('%array_functions(LLVMBasicBlockRef,LLVMBasicBlockRefArray)\n')
		if contents.find('LLVMExecutionEngineRef *') >= 0:
			out.write('%array_functions(LLVMExecutionEngineRef,LLVMExecutionEngineRefArray)\n')
		if contents.find('char **') >= 0:
			out.write('%array_functions(char *,StringArray)\n')

		# Add SWIG imports for includes of LLVM header files.
		for m in re.finditer('#include *"llvm-c/([^"]*)\.h"', contents):
			out.write('\n%import "' + m.group(1) +'.i"\n')
		if name == 'TargetMachine':
			out.write('%import "Target.i"\n')

		out.write(contents)
		out.close()

	# Because some *.i files import others, run SWIG on them only after
	# generating them all.
	print
	for header in headers:
		name = os.path.basename(header).rpartition('.')[0]
		relpath = os.path.normpath('./' + last_token(llvm_c_dir,'/')) # + '/' + os.path.dirname(header)[len(llvm_c_dir):] + '/')
		cwd = os.getcwd()
		os.chdir(relpath)
		runprog("swig",["-package org.jllvm.bindings -java",name + '.i'])
		os.chdir(cwd)
		
if __name__ == '__main__':
	generate_swig_interfaces(sys.argv[1])
	runprog("cp",["bindings/.gitignore", "bindings/CMakeLists.txt", "bindings/FindLLVM.cmake", "llvm-c"])
