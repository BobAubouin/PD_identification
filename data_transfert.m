File_list = dir("Data_Mat_files_rug_improved/");

for i=1:length(File_list)
    if length(File_list(i).name) > 5
        load("Data_Mat_files_rug_improved/" + File_list(i).name)
        

        writetimetable(TT_resampled, "Patient_2_data.csv")
    end


end