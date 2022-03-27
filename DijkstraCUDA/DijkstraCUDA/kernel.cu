/*..............................................................................................................................*/
// Author: Dinu Ion George
// Date:
// Remarks:
/*..............................................................................................................................*/

/* Local application / library specific imports */
#include "Dijkstra.cuh"

/*Core main function*/
int main()
{
    /*Store the path to the input test cases, and output generated data*/
    getParentDirectoryPath();

    strcpy(inputTestsPath, parrentDirectoryPath);
    strcat(inputTestsPath, "\\input_files\\test_.in");

    strcpy(outputTestsPath, parrentDirectoryPath);
    strcat(outputTestsPath, "\\CUDA_output_files\\test_.out");

    startTests();
    return 0;
}

