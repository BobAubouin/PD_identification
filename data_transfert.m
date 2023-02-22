File_list = dir("Data_Mat_files_rug_improved/");

for i=1:length(File_list)
    if length(File_list(i).name) > 5
        load("Data_Mat_files_rug_improved/" + File_list(i).name)
        [~, end_word] = strtok(File_list(i).name, '_');
        end_word = strtok(end_word, '.');
        Patient_id = end_word(3:end);
        disp(Patient_id)
        TT_bis = Clean_ttable(TT_bis);
        TT_prop = Clean_ttable(TT_prop);
        TT_remi = Clean_ttable(TT_remi);
        TT_nsn2 = Clean_ttable(TT_nsn2);
        TT_nsn  = Clean_ttable(TT_nsn);
        plot(TT_bis.Time, TT_bis.BIS1,'b')
        TT_resampled = synchronize(TT_bis, TT_prop, TT_remi, TT_nsn2, TT_nsn ,'regular','linear','TimeStep',seconds(5),'EndValues',-1);
        hold on
        plot(TT_resampled.Time, TT_resampled.BIS1,'*r');
        saveas(gcf,"outputs/BIS patient " + Patient_id + ".png");
        close

        writetimetable(TT_resampled, "data/raw/Patient_" + Patient_id + "_data.csv");
        save("outputs/" + File_list(i).name, "TT_resampled");
        
    end


end

function TT_clean = Clean_ttable(TT)
%Clean Timetables
%  function based on https://nl.mathworks.com/help/matlab/matlab_prog/clean-timetable-with-missing-duplicate-or-irregular-times.html
%% clean & sorted
    goodValuesTT = rmmissing(TT); %clean NAN Values
    sortedTT = sortrows(goodValuesTT);
    uniqueRowsTT = unique(sortedTT);
    dupTimes = sort(uniqueRowsTT.Time);
    tf = (diff(dupTimes) == 0);
    dupTimes = dupTimes(tf);
    dupTimes = unique(dupTimes);

    if ~isempty(dupTimes)
        uniqueTimes = unique(uniqueRowsTT.Time);
        TT_clean = retime(uniqueRowsTT,uniqueTimes,'firstvalue');
    else
        TT_clean = uniqueRowsTT;
    end
end
