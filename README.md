# PD_identifiction
Code to indentify the PD parameter of the PRopofol Remifentanil to BIS interaction on clinical data.

## Usage
First, all the patient data must be convert to*.csv* files and placed in the *data* folder with the name *patient_i_data.csv* where *i* is the patient id.
To transfert data from matlab to csv the matlab script *data_transfert.m* can be used.

Then the needed packages for python script must be installed with the command line:
'''
pip install -r requirements.txt
'''

Then the script *dataset_identification.py* can be launched. The output table is stored as a *.csv* file in the output folder.

## Files description
* *dataset_identification.py* : PD identification of the full dataset.
* *test_models.py* : PD identification for different PK model (Eleveld or Scnhider and Minto" for the Propofol to BIS system.
* *test_miso_siso.py* : Test the identification in two phase (Propo to Bis and Remi to BIS) versus in one phase (Propo-Remi to BIS)
* *TCI_control.py* : include the code to reproduce the controller inside the TCI device.
* *data_transfert.m* : Matlab file to trasfert the data from .mat files to *.csv* files.


## License

_GNU General Public License 3.0_

## Project status
Stable, code running on the 70 patients database.

## Author
Bob Aubouin--Paitault
