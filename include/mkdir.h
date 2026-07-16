#ifndef __MKDIR__
#define __MKDIR__

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif
#include <string.h>
#include <stdio.h>

// Named make_dir_recursive (not _mkdir) to avoid colliding with the Windows
// CRT's own _mkdir(), which this function calls on that platform.
static void make_dir_recursive(const char *dir) {
    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
#ifdef _WIN32
            _mkdir(tmp);
#else
            mkdir(tmp, S_IRWXU);
#endif
            *p = '/';
        }
#ifdef _WIN32
    _mkdir(tmp);
#else
    mkdir(tmp, S_IRWXU);
#endif
}

#endif __MKDIR__