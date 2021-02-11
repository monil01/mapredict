// For testing and debugging a transactional NVL-C application's ability to
// recover after premature termination (due to power loss, for example),
// this library provides a simple API to define "kill positions".  A kill
// position is a source location within the NVL runtime or application code
// at which premature application termination can be simulated selectively
// via environment variables.
//
// Probably the easiest way to discover cases in which your transactional
// NVL-C application is not able to recover after premature termination is
// to take a shotgun approach.  That is, write a script that runs that
// application in a loop and that, for each iteration, (1) sends it a
// SIGQUIT signal (Java's Process.destroy method uses this signal) after a
// random period of time and then (2) runs the application again in recovery
// mode.  Once the script finds a case where recovery fails, you can use a
// debugger and the coredump produced by SIGQUIT to figure out where in the
// control flow the SIGQUIT actually arrived.  However, while trying to
// debug that failure, you'll likely wish to script more precise
// experiments, and eventually you'll probably want to set up deterministic
// test cases.  This library helps with those scenarios.
//
// To use this library, follow the procedure below.  This procedure assumes
// your application has deterministic control flow so that run-time
// encounters with kill positions can be numbered deterministically.
//
// 1. Instrument the source code:
//
//   Add KILLPOS() calls, one per source line, at positions where you wish
//   to selectively cause application termination within the NVL runtime
//   and/or your application.  At the end of your application (probably at
//   the end of your main function), print the result of a
//   killpos_getHitCount() call.  Of course, include killpos.h to access the
//   killpos API.  Recompile the NVL runtime and/or application.
//
// 2. Profile your application without premature termination:
//
//   Run your application and record the value printed from
//   killpos_getHitCount().  This is the total number of kill position
//   encounters at run time.  If your application control flow is
//   deterministic as required, it should always have the same result until
//   you enable premature termination, as in the following steps.
//
// 3. Search for an unrecoverable premature termination:
//
//   Run your application in a loop, where each iteration has two steps:
//
//     Step a: Run your application with the environment variable KILLPOS
//     set to an integer in the interval [0, killpos_getHitCount()).  You
//     can choose the integer randomly, or your loop can progress through
//     every possible integer in the interval.  Your application will
//     terminate at the kill position encounter selected by KILLPOS, and it
//     will print the source file and line number of the kill position.
//     In order to terminate, it calls abort, which produces a core dump
//     if you've enabled core dumps (ulimit -c unlimited).
//
//     Step b: Run your application in recovery mode with the environment
//     variable KILLPOS unset.  If recovery fails, record the source file
//     and line number printed by substep a.
//
// 4. Study failure:
//
//   Once step 3 finds a failure, study it.  To do so, repeat step 3 with
//   two differences.  First, in substep a, always set the environment
//   variable KILLPOS to the integer that previously produced the failure in
//   step 3.  Second, also in substep a, set the environment variable
//   KILLPOS_LINE to "file:line" where file and line were reported in step
//   3b.
//
//   Your application will now count and print every encounter of the kill
//   position that produced the failure, and it will still terminate at
//   the same encounter of that kill position.  To terminate the application
//   at different encounters of the same kill position, leave KILLPOS unset,
//   and set KILLPOS_LINE to "file:line:i" where i is the encounter index.

#ifndef KILLPOS_H
#define KILLPOS_H

#define KILLPOS() killpos(__FILE__, __LINE__)
void killpos(const char *file, long line);
long killpos_getHitCount();

#endif
