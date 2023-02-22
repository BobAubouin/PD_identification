# PD_identifiction
Code to indentify the PD parameter of the PRopofol Remifentanil to BIS interaction on clinical data.

## Usage
First, all the patient data must be convert to *.csv* files and placed in the *data/raw* folder with the name *patient_i_data.csv* where *i* is the patient id. All the *.txt* logfiles must be placed in the folder *data/LogFiles*.

To transfert data from Matlab to *.csv* the matlab script *data_transfert.m* can be used. Data folder must also contains a file name *patients_fiche.csv* which contains the patient informations with the columns *No.* (patient id), *Sex* (F or M), *Age* (yr), *Height* (cm) and *Weight* (kg).

The needed packages for python script must be installed with the command line:
```python
pip install -r requirements.txt
```
Before launching porpofol identification script, the script *process_raw_data.py* must be launched.

Finally, the identification scripts can be launched. The output table is stored as a *.csv* file in the output folder.

## Files description
* *identif_propo.py* : solve the following identification problem for each patient:
```math
\tag{1}
\min_x \sum_i^N \left(BIS(t_i) - (E_0 - E_{max} \frac{C_e(t_i)^\gamma}{C_{50}^\gamma + C_e(t_i)^\gamma})\right)^2
```

with $E_0$ fix to the mean value over the first 10 seconds and $x = (C_{50}, \gamma, E_{max} )$.
* *identif_propo_Emax=E0.py*: also solve problem (1) for all the patient but fixing $E_{max}=E_0$ with $x = (C_{50}, \gamma)$
* *identif_prop_remi.py*: solve the following identification problem for all the patient:
```math
\tag{2}
\min_x \sum_i^N \left(BIS(t_i) - pred(t_i)\right)^2
```
with:
```math
pred(t) = E_{0} - E_{max} \frac{U(t)^\gamma}{1 + U(t)^\gamma}
```

```math
U(t) = \frac{U_p(t) + U_r(t)}{1 - \beta \theta(t) + \beta \theta(t)^2}
```
```math
U_p(t) = \frac{C_{ep}(t)}{C_{50p}}; \; U_r(t) = \frac{C_{er}(t)}{C_{50r}}; \; \theta(t) = \frac{U_p(t)}{U_p(t)+U_r(t)}
```
and $x = (C_{50p}, C_{50r}, \gamma, \beta, E_{max} )$.
* *identif_prop_remi_Emax=E0.py*: also solve problem (2) for all the patient but fixing $E_{max}=E_0$ with $x = (C_{50p}, C_{50r}, \gamma, \beta,)$
* *stats.py*: display useful statitistic about the results of the identifications.
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
