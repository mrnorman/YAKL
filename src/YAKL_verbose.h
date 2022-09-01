
#pragma once

namespace yakl {

  /** @private */
  inline void verbose_inform(std::string prefix, std::string label = "", std::string suffix = "") {
    #ifdef YAKL_VERBOSE
      // Form the output
      std::string output = std::string("*** [YAKL_VERBOSE] ") + prefix;
      if (label != "") output += std::string(" (label: \"") + label + std::string("\")");
      if (suffix != "") output += std::string(";  ") + suffix;

      // Get MPI rank
      int rank = 0;
      #ifdef HAVE_MPI
        int is_initialized;
        MPI_Initialized(&is_initialized);
        if (is_initialized) { MPI_Comm_rank(MPI_COMM_WORLD, &rank); }
      #endif

      // Write to file
      #ifdef YAKL_VERBOSE_FILE
        std::ofstream myfile;
        std::string fname = std::string("yakl_verbose_output_task_") + std::to_string(rank) + std::string(".log");
        myfile.open(fname , std::ofstream::out | std::ofstream::app);
        myfile << output << std::endl;
        myfile.close();
      #endif

      // Write to stdout for task 0
      if (rank == 0) std::cout << output << std::endl;
    #endif
  }

}


