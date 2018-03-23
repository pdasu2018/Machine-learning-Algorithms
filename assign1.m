clc
file = 'SkillCraft1_Dataset.csv';
y = xlsread(file)

GapPAC = y(:,13);
ActPAC = y(:,15);
figure()
histogram(GapPAC)
title(' Frequency Histogram of GapBetweenPACs')
grid on ;

figure()
histogram(ActPAC);
title(' Frequency Histogram of ActionsInPACs')
grid on;
figure()
scatter(y(:,7), y(:,8),'filled', 'r')
xlabel ('SelectByHotKeys');
ylabel ('AssignToHotKeys');
title('Scatter plot of SelectByHotKeys v/s AssignToHotKeys');

figure()
scatter(y(:,12), y(:,13), 'g')
xlabel ('NumberOfPACs');
ylabel ('GapBetweenPACs');
title('Scatter  plot of NumberOfPACs v/s GapBetweenPACs')
A = y(:, 6:20)
U = corrcoef(A)
max = 0 
for i = 1:15
    for j = 1:15
        if ((j~= i) &&( U(i,j) >max ))
            max = U(i,j)
            row = i 
            col = j 
        end 
    end 
end 

min = 0 
for i = 1:15
    for j = 1:15
        if ((j~= i) &&( U(i,j) <min ))
            min = U(i,j)
            row_min = i 
            col_min = j 
        end 
    end 
end 
save('pcc.mat','U');        
figure()
scatter(y(:,5+col_min), y(:,5+row_min), 'r')
xlabel ('ActionLatency');
ylabel ('NumberOfPACs');
title('Scatter plot for minimum correlation');

figure()
scatter(y(:,5+col), y(:,5+row), 'r')
title('scatter plot for maximum correlation');
xlabel ('SelectByHotkeys');
ylabel ('APM');
