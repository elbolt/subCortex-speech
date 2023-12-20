#!/bin/bash

#######################################################################################
# Running scripts for AN rates extraction while switching between conda environments. #
#######################################################################################

# Description: This script controls a workflow involving two different conda environments.
# It executes my Python scripts in each environment in the order I need it.
# I apologize for the sloppy implementation, it was the best solution I could come up with at the time.

ENV_NAME="neuro"
ENV_AN_NAME="linfeatures"
WAVE="normal"  # change to "inverted" to run the pipeline on the inverted speech wave

# Activate conda environment "environment"
mamba activate $ENV_NAME

# Run `prepare_model_input.py`
echo "Running prepare_model_input.py ..."
python prepare_model_input.py -i $WAVE

# Deactivate environment and activate "an_model_environment"
mamba deactivate $ENV_NAME
mamba activate $ENV_AN_NAME

# Run the program `run_model.py``
echo "Running run_model.py ..."
python run_model.py -i $WAVE

# Deactivate an_model_environment and switch back to usual one
mamba deactivate $ENV_AN_NAME
mamba activate $ENV_NAME

# Run the python program "process_model_output.py"
echo "Running process_model_output.py ..."
python process_model_output.py -i $WAVE

echo "AN rates extraction complete."
mamba deactivate
