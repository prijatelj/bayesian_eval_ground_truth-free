/**
 * A Command Line Interface (CLI) wrapper for using the CEKA inference methods
 * to be used in testing the annotation aggregation / truth inference methods.
 */
package psych_metric;

import java.util.

import ceka.consensus.GTIC;
import ceka.core.Dataset;
import ceka.core.Example;

// TODO read in the dataset in proper format for CEKA, a converter script will be needed and those versions saved separately probably.
private void static loadDataset(, String outputPath, String inferenceMethodIds[]){


    return new ceka.core.Dataset()
}

// TODO given dataset and set of CEKA inference method ids, run the methods, save the outputs as desired.
private void static runInference(ceka.core.Dataset dataset, String outputPath, String inferenceMethodIds[]){


    for (int i=0; i < inferenceMethodsIds.length; i++){
        // run one of the inference methods on the dataset

        // save the results to the correct location based on given root directory.
    }

}

public void static main(String args[]){
    // Load the dataset as ceka.core.Dataset
    dataset = loadDataset(args[0]);

    // Run inference methods and save results for each.
    runInference(dataset, args[1], Arrays.stream(args).skip(2).toArray(String[]::new));
}
