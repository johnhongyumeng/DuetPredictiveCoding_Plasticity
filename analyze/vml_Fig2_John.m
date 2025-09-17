clearvars -except agg* proj*
close all
%%
pp=fieldnames(aggregateData);
iP=1;
corr_thr=0.02;
%%
parameters_main;
sumIDX=35:45;
normalization=19:29;
TP = 1;
cl=[-.2 .2];
grayscale=.97;

pp=fieldnames(aggregateData);
% ?? for iP=7
for iP=1   
    IDX = aggregateData.(pp{iP}).expType{TP};
    MM = aggregateData.(pp{iP}).avgMM{TP};
    MMR= aggregateData.(pp{iP}).avgMMRandom{TP};
    CM = aggregateData.(pp{iP}).corrM{TP};
    CP = aggregateData.(pp{iP}).corrP{TP};
    FF = aggregateData.(pp{iP}).ROIinfo{TP};
    %MM=MM-MMR;
    mm=mean(MM(meanPost,:))-mean(MM(meanPre,:));
    tmp=aggregateData.(pp{iP}).avgMM{TP}-aggregateData.(pp{iP}).avgMMRandom{TP};
    mm=mean(tmp(meanPost,:))-mean(tmp(meanPre,:));

    FF=FF(:,1);

    for type=0:1
    % type=1;
        figure(type+1)
        
        %mm=mean(MM(sumIDX,:));
        actIDX=FF>4;
        realP=~isnan(CP);
        realM=~isnan(CM);
        
        d = -0.6:0.025:0.6;
        
        MIDX_CT=IDX==type & actIDX & realP & realM;
        
        
        aCT=CM(MIDX_CT);
        bCT=CP(MIDX_CT);
        fprintf('Corr: %0.3f',corr(aCT,bCT))
        cCT=mm(MIDX_CT);
        m=length(cCT);
        n = fix(m/2);
        x = n~=(m/2); 
        %r = [(0:1:n-1)/n,ones(1,n+x)]; 
        %g = [(0:1:n-1)/n,ones(1,x),(n-1:-1:0)/n]; 
        %bl = [ones(1,n+x),(n-1:-1:0)/n];
        r = grayscale*[(0:1:n-1)/n,ones(1,n+x)];
        %r(r<.5)=.5;
        bl = grayscale*[(0:1:n-1)/n,ones(1,x),(n-1:-1:0)/n]; 
        g = grayscale*[ones(1,n+x),((n-1:-1:0)/n)];
        g(g<.65)=.65;
        
        cmap = [r(:),g(:),bl(:)];
        
        [~,sidx]=sort(cCT);
        %cCT(cCT<0)=2*cCT(cCT<0);
        sidx=1:length(cCT);
        
        % subplot(1,2,1+type)
        hold on
        scatter(bCT(sidx),aCT(sidx),15,cCT(sidx),'filled','MarkerEdgeColor', 'k')
    
        % Compute angles in radians
        % norm_bCT=bCT(sidx)-mean(bCT(sidx));
        % norm_aCT=aCT(sidx)-mean(aCT(sidx));
        norm_bCT=bCT(sidx);
        norm_aCT=aCT(sidx);
        dist_all= sqrt(norm_bCT.^2+norm_aCT.^2);
    
    
        set(gcf, 'Position', [100, 100, 400, 300]); % [x, y, width, height]
    
        set(gca,'CLim',cl,'XTick',0,'YTick',0)
        colormap(cmap);
        % colorbar
        axis image
        xlim([-.7 .7])
        ylim([-.7 .7])
        hold on
        grid on
    
        file_name = ['./Figures/' pp{1} '_type_' num2str(type),'_scattor' ];  % Use the first field name for naming
        % Save the figure in different formats
        saveas(gcf, [file_name, '.fig']);  % Save as MATLAB .fig file
        saveas(gcf, [file_name, '.svg']);  % Save as SVG
        
        sig_thr=(norm_bCT.^2+norm_aCT.^2)>corr_thr;
        % sig_thr=(norm_bCT.^2+norm_aCT.^2)<corr_thr;
        num_cell=sum(sig_thr);
        norm_aCT_sig=norm_aCT(sig_thr);
        norm_bCT_sig=norm_bCT(sig_thr);
        % theta_corr = atan2(norm_bCT(sig_thr),norm_aCT(sig_thr) );
        theta_corr = atan2(norm_bCT,norm_aCT );
        % figure(666) % Find the optimal threshold.
        % clf
        % 
        % 
        % % Define more bins for higher resolution
        % num_bins = 50; % Increase number of bins
        % bin_edges = logspace(log10(min(dist_all)), log10(max(dist_all)), num_bins); % Log-spaced bins
        % 
        % % Plot histogram
        % histogram(dist_all, 'BinEdges', bin_edges, 'Normalization', 'probability', ...
        %           'FaceColor', 'blue', 'EdgeColor', 'black', 'FaceAlpha', 0.7);
        % 
        % set(gca, 'XScale', 'log');
        % ylabel('Probability Density');
        % title('Histogram of distance (Log-Scale X-Axis)');
        % 
        % % Set axes properties
        % set(gca, 'FontSize', 12);
    
    
        % Adjust angles that are less than -3/4 * pi
        theta_corr(theta_corr < -3/4 * pi) = theta_corr(theta_corr < -3/4 * pi) + 2 * pi;
        
        % Create figure
        bin_width = pi / 10.0;
        bin_edges = -3 * pi / 4 : bin_width : (5/4 * pi + bin_width); % Define bins
        % Compute weighted histogram manually
        [counts, edges] = histcounts(theta_corr, bin_edges, 'Normalization', 'probability');
        weighted_counts = accumarray(discretize(theta_corr, bin_edges), dist_all, [length(counts), 1], @sum);
        weighted_counts = weighted_counts/sum(weighted_counts) ;   
    

        % Use bin_centers as the "data", weighted by weighted_counts
        bin_centers = (edges(1:end-1) + edges(2:end)) / 2;

       
        % Initialize pseudo sample container
        theta_pseudo = [];
        sum_pseudo_weight=0;
        sum_weight= sum(dist_all);
        rng(1);  % Set seed for reproducibility
        % Loop through each point
        for i = 1:length(theta_corr)
            % Repeat theta_corr(i) 100 times with inclusion probability dist_all(i)
            % if theta_corr(i)/pi-1/4>3/4 || theta_corr(i)/pi-1/4<-3/4
            %     continue
            % end
            if theta_corr(i)/pi-1/4>1/2 || theta_corr(i)/pi-1/4<-1/2 
                continue
            end
            if dist_all(i)<0.1
                continue
            end
            sum_pseudo_weight=sum_pseudo_weight+dist_all(i);
            n_repeat = sum(rand(1, 200) < dist_all(i));  % Number of times to include this point
            theta_pseudo = [theta_pseudo; repmat(theta_corr(i), n_repeat, 1)];
        end
        % Get BIC scores
        Neff = (sum(dist_all))^2 / sum(dist_all.^2);
        options= statset('MaxIter', 1000);
        gm1 = fitgmdist(theta_pseudo-pi/4, 1, 'RegularizationValue', 1e-5,'Options', options);
        gm2 = fitgmdist(theta_pseudo-pi/4, 2, 'RegularizationValue', 1e-5,'Options', options);

        bic1 = gm1.BIC;
        bic2 = gm2.BIC;
        aic1 = gm1.AIC;
        aic2 = gm2.AIC;        
        logL1 = gm1.NegativeLogLikelihood * -1;
        logL2 = gm2.NegativeLogLikelihood * -1;
        k1 = gm1.NumComponents;
        k2 = gm2.NumComponents;
        % bic1 = -2 * logL1 + k1 * log(Neff);
        % bic2 = -2 * logL2 + k2 * log(Neff);

        bic1_eff = -2 * logL1 + k1 * log(Neff);
        bic2_eff = -2 * logL2 + k2 * log(Neff);

        fprintf('BIC (1 comp): %.2f\n', bic1);
        fprintf('BIC (2 comp): %.2f\n', bic2);        
        fprintf('AIC (1 comp): %.2f\n', aic1);
        fprintf('AIC (2 comp): %.2f\n', aic2); 
        fprintf('BIC eff (1 comp): %.2f\n', bic1_eff);
        fprintf('BIC eff (2 comp): %.2f\n', bic2_eff);        


        figure(type+21)
        clf
        histogram(theta_pseudo-pi/4, 'Normalization', 'pdf', 'BinWidth', 0.05); 
        set(gcf, 'Position', [200,300,560, 420]); % [left, bottom, width, height]
        hold on;
        xgrid = linspace(min(theta_pseudo-pi/4), max(theta_pseudo-pi/4), 200)';
        y1 = pdf(gm1, xgrid);
        y2 = pdf(gm2, xgrid);
        plot(xgrid, y1, 'r-', 'LineWidth', 2);
        plot(xgrid, y2, 'b-', 'LineWidth', 2);
        % Plot vertical lines at component means
        means1 = gm1.mu;
        means2 = gm2.mu;
        
        for m = 1:length(means1)
            xline(means1(m), 'r--', 'LineWidth', 1.5);
        end
        
        for m = 1:length(means2)
            xline(means2(m), 'b--', 'LineWidth', 1.5);
        end

        legend('Data', '1-comp GMM', '2-comp GMM');
        xticks([-pi / 2, 0, pi/2]);
        xticklabels({'-\pi/2', '0', '\pi/2'});
        set(gca, 'FontSize', 18, 'XMinorTick', 'on'); % Adjust font size if needed

        % Add BIC difference in the title
        BIC_diff = gm1.BIC - gm2.BIC;  % Positive value favors gm2 (2-component)
        if BIC_diff > 0
            preferred = '2-comp GMM';
        else
            preferred = '1-comp GMM';
        end
        box off
        title(sprintf('Gaussian Mixture Model Fit (ΔBIC = %.2f, Preferred: %s)', BIC_diff, preferred));
        file_name = ['./Figures/' pp{1} '_type_' num2str(type),'_fittedGMM' ];  % Use the first field name for naming
        % Save the figure in different formats
        saveas(gcf, [file_name, '.fig']);  % Save as MATLAB .fig file
        saveas(gcf, [file_name, '.svg']);  % Save as SVG



        figure(type+11)
        clf
        hold on
        % bar((bin_edges(1:end-1)+bin_edges(2:end))-pi/4, weighted_counts/bin_width, 'FaceColor', 'blue', 'EdgeColor', 'black', 'FaceAlpha', 0.7, 'BarWidth', 1);
        bar(bin_edges(1:end-1)-pi/4+bin_width/2, weighted_counts/bin_width, 'FaceColor', 'blue', 'EdgeColor', 'black', 'FaceAlpha', 0.7, 'BarWidth', 1);

        set(gcf, 'Position', [200,300,300, 225]); % [left, bottom, width, height]

        xgrid=linspace(-pi, pi,200)';
        xgrid_width=2*pi/200;
        y1 = pdf(gm1, xgrid);
        y2 = pdf(gm2, xgrid);
        

        % y1_norm=y1;
        % y2_norm=y2;

        y1_norm=y1*sum_pseudo_weight/sum_weight;
        y2_norm=y2*sum_pseudo_weight/sum_weight;

        % plot(xgrid, y1_norm, 'k-', 'LineWidth', 2);
        % plot(xgrid, y2_norm, 'r-', 'LineWidth', 2);    
        box off
        % % Plot histogram
        % histogram(theta_corr, 'BinEdges', bin_edges, 'Normalization', 'probability', ...
        %           'FaceColor', 'blue', 'EdgeColor', 'black', 'FaceAlpha', 0.7,'Weights',dist_all(sig_thr));

        % Customize x-ticks
        % xticks([-3 * pi / 4, -pi / 4, pi / 4, 3/4 * pi, 5/4 * pi]);
        xticks([-pi, -pi / 2, 0, pi/2, pi]);
        xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

        % Set axis labels (optional)

        hold off
        % Set axes properties
        set(gca, 'FontSize', 12, 'XMinorTick', 'on'); % Adjust font size if needed
        file_name = ['./Figures/' pp{1} '_type_' num2str(type),'_hist' ];  % Use the first field name for naming
        % Save the figure in different formats
        saveas(gcf, [file_name, '.fig']);  % Save as MATLAB .fig file
        saveas(gcf, [file_name, '.svg']);  % Save as SVG
        % Save raw data to .mat file
        % file_name = ['./Figures/' pp{1} '_type_' num2str(type) ];  % Use the first field name for naming

        % save([file_name, '_data.mat'], 'bCT', 'aCT');


    end 
end


% % Silverman test.
% [p, h_crit, h_boot] = silverman_test(theta_pseudo, dist_all,200);
% 
% % Optional: visualize bootstrap distribution of critical bandwidths
% figure;
% histogram(h_boot, 'Normalization', 'pdf');
% hold on;
% xline(h_crit, 'r--', 'LineWidth', 2);
% xlabel('Critical Bandwidth');
% ylabel('PDF');
% title(sprintf("Silverman's Test p = %.3f", p));