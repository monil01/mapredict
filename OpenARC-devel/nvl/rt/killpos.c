#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool killposInit = false;
static long killposHitCount = 0;
static long killposHitSelect = -1;

static char *killposFile = NULL;
static long killposLine;
static long killposLineHitCount = 0;
static long killposLineHitSelect;

void killpos(const char *file, long line) {
  if (!killposInit) {
    const char *killposEnv = getenv("KILLPOS");
    fprintf(stderr, "env KILLPOS: %s\n", killposEnv);
    if (killposEnv) {
      killposHitSelect = atoi(killposEnv);
      fprintf(stderr, "killposHitSelect = %ld\n", killposHitSelect);
    }
    const char *killposLineEnv = getenv("KILLPOS_LINE");
    fprintf(stderr, "env KILLPOS_LINE: %s\n", killposLineEnv);
    if (killposLineEnv) {
      const char *colon0 = strchr(killposLineEnv, ':');
      size_t len = colon0 - killposLineEnv;
      killposFile = malloc((len+1) * sizeof *killposFile);
      strncpy(killposFile, killposLineEnv, len);
      killposFile[len] = '\0';
      killposLine = atoi(colon0+1);
      const char *colon1 = strchr(colon0+1, ':');
      killposLineHitSelect = colon1 ? atoi(colon1+1) : -1;
      fprintf(stderr, "killposFile = %s\n", killposFile);
      fprintf(stderr, "killposLine = %ld\n", killposLine);
      fprintf(stderr, "killposLineHitSelect = %ld\n",
              killposLineHitSelect);
    }
    killposInit = true;
    fflush(stderr);
  }
  bool killNow = false;
  if (killposFile && killposLine == line && !strcmp(killposFile, file)) {
    killNow = killposLineHitSelect == killposLineHitCount;
    fprintf(stderr, "killpos %ld at %s:%ld:%ld: %s...\n",
            killposHitCount, file, line, killposLineHitCount,
            (killNow ? "killng" : "skipping"));
    fflush(stderr);
    ++killposLineHitCount;
  }
  if (killposHitCount == killposHitSelect) {
    killNow = true;
    fprintf(stderr, "killpos %ld at %s:%ld: killing...\n",
            killposHitCount, file, line);
    fflush(stderr);
  }
  ++killposHitCount;
  if (killNow)
    abort();
}

long killpos_getHitCount() {
  return killposHitCount;
}
