# Copyright (c) 2021-2022  The University of Texas Southwestern Medical Center.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted for academic and research use only (subject to the
# limitations in the disclaimer below) provided that the following conditions are met:

#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.

#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.

#      * Neither the name of the copyright holders nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

from navigate.tools.decorators import AcquisitionMode
from navigate.model.features.feature_related_functions import OpmAcquisition


@AcquisitionMode
class OpmAcquisitionMode:
    def __init__(self, name):
        self.acquisition_mode = name

        self.feature_list = [{"name": OpmAcquisition}]

    def prepare_acquisition_controller(self, controller):
        """Prepare acquisition in controller side

        Parameters
        ----------
        controller : object
            navigate controller
        """
        microscope_setting_dict = controller.configuration["experiment"][
            "MicroscopeState"
        ]
        microscope_setting_dict["waveform_template"] = "Confocal-Projection"

        # backup stack cycling mode
        self.stack_cycling_mode = microscope_setting_dict["stack_cycling_mode"]

        microscope_setting_dict["stack_cycling_mode"] = (
            "per_stack"
            if microscope_setting_dict["conpro_cycling_mode"] == "per_stack"
            else "per_z"
        )

        # scan range
        if microscope_setting_dict["scanrange"] <= 0:
            microscope_setting_dict["scanrange"] = 1
            print("Scan range shouldn't be 0! Please set the scan range value!")

    def end_acquisition_controller(self, controller):
        """Cleanup in controller side after acquisition

        Parameters
        ----------
        controller : object
            navigate controller
        """
        controller.configuration["experiment"]["MicroscopeState"][
            "waveform_template"
        ] = "Default"

        # reset cycling mode
        controller.configuration["experiment"]["MicroscopeState"][
            "stack_cycling_mode"
        ] = self.stack_cycling_mode

    def end_acquisition_model(self, model):
        """Cleanup in model side after acquisition"""
        for stage, _ in model.active_microscope.stages_list:
            if type(stage).__name__ == "GalvoNIStage":
                stage.switch_mode("normal")

    def update_saving_config(self, model):
        """Update saving configuration (set the shaping metadata)

        Parameters
        ----------
        model : object
            navigate model
        """
        return {"z": model.configuration["experiment"]["MicroscopeState"]["n_plane"]}
