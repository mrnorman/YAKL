
#pragma once

namespace yakl {

  inline void verbose_inform(std::string prefix, std::string label = "", std::string suffix = "") {
    #ifdef YAKL_VERBOSE
      // Form the output
      std::string output = prefix;
      if (label != "") output += std::string(" (label: \"") + label + std::string("\")");
      if (suffix != "") output += std::string(";  ") + suffix;
      output += std::endl + std::endl;
      #ifdef YAKL_VERBOSE_FILE
        // Get the MPI rank
        int rank;
        #ifdef HAVE_MPI
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #else
          rank = 0;
        #endif
        // Write to file
        ofstream myfile;
        std::string fname = std::string("yakl_verbose_output_task_") + std::to_string(rank) + std::string(".log");
        myfile.open(fname);
        myfile << output;
        myfile.close();
      #endif
      // Write to stdout for task 0
      if (yakl_mainproc()) std::cout << output;
    #endif
  }

}


