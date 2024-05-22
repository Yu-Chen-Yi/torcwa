clear;clc;close all;
% TA_Image = double(imread("dark.bmp"));
Ny = 500;
Nx = 1000;
TA_Image = zeros(Nx,Ny);
a = 1;
for i = -6:6
    for j = -3:3
        TA_Image(round(Nx/2)+1+i*a,round(Ny/2)+1+j*a) = 1;
    end
end
i = 0;
j = 0;
TA_Image(round(Nx/2)+1+i*a,round(Ny/2)+1+j*a) = 0;
%     TA_Image(26,26) = 1;

figure(2)
pcolor(TA_Image);shading flat;
TA = kron(TA_Image,ones(1,1));
% Initalized DOE Device : E0 * exp(1i*phase)
TA0 = 1*exp(1i*2*pi*rand(size(TA,1),size(TA,2)));
level = 2;
MES = [];
% imagesc(TA)
%
for i = 1:100
    DOE0 = ifftshift(ifft2(fftshift(TA0)));
    DOE0_phase = angle(DOE0)/2/pi;
    % DOE0_phase2 = round(DOE0_phase*(level-1));
    A = rem(DOE0_phase,1/level);
    filter1 = A > (1/level)/2*0.6;
    filter2 = A < -(1/level)/2;
    A = A + filter1*-(1/level)+filter2*(1/level);
    DOE0_phase = DOE0_phase - rem(A,1/level);
    DOE0_phase = DOE0_phase*0.813;
    % DOE0_phase = round(DOE0_phase);
    % DOE0_phase = DOE0_phase/level;

    % imagesc(DOE0_phase)
    DOE1 = 1*exp(1i*2*pi*DOE0_phase);
    TA0 = ifftshift(fft2(fftshift(DOE1)));
    TA0_phase = angle(TA0);
    I0 = abs(TA0).^2/sum(sum(abs(TA0).^2));
    MES =[MES sum(sum(abs((I0 - TA)).^2))/length(I0(:))];
    TA0 = TA.*exp(1i*TA0_phase);
end
%%
% I0 = I0(496:506,496:506);
% eff = sum(sum(I0(:,2:10)))/sum(sum(I0));
% uA = unique(nonzeros(I0(:,2:10))); %does sorting and remove duplicates
% small2distinct = uA(2);
% uni = small2distinct/max(max(I0(:,2:10)));
% sum(sum(I0(496:506,497:505)))/sum(sum(I0))
% xor = -5:1:5;
% yor = -5:1:5;
% A = DOE0_phase*-pi;
% DOE1 = 1*exp(1i*2*pi*A);
% TA0 = fft2(DOE1);
% I0 = abs(TA0).^2/max(max(abs(TA0).^2));
img = abs(DOE0_phase)/max(max(abs(DOE0_phase)));
imwrite(img,"himax_7x13.png")
% h1 = figure(1);
% set(h1,'Name','Image','color','w','numberTitle','off','Units','normalized','Position',[1 0 1 1]);
% subplot(1,3,1)
% imagesc(abs(DOE0_phase));axis equal;axis tight;colorbar;
% xlabel("\itx axis (pixel)"); ylabel("\ity axis (pixel)"); title("Level="+level+"; Phase of DOE");
% % h2 = figure(2);
% % set(h2,'Name','Image','color','w','numberTitle','off','Units','normalized','Position',[1 0 0.5 1]);
% subplot(1,3,2)
% imagesc(xor,yor,I0);axis equal;axis tight;colorbar;
% % set(gca,'colorscale','log')
% xlabel("\itx order (order)"); ylabel("\ity order (order)"); title("Level="+level+"; Target Image");
% % h3 = figure(3);
% % set(h3,'Name','RMS','color','w','numberTitle','off','Units','normalized','Position',[0 0 0.5 0.6]);
% subplot(1,3,3)
% plot(MES,'linewidth',2)
% xlabel("\itIteration"); ylabel("\itRMS"); title("");
% set(gca,'Fontsize',20);
% axis square;