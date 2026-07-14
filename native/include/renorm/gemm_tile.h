#pragma once

namespace renorm
{

struct GemmTile
{
    int ii;
    int jj;
    int kk;

    int iEnd;
    int jEnd;
    int kEnd;
};

}