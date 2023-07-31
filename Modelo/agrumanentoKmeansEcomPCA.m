clear 
clc
close all
%% --------------------- Base de dados ------------------------------------
% Ler a base de dados
Customrs = readtable('Customers.csv', 'VariableNamingRule', 'preserve');
% Apagando o atributo de indices das amostras
Customrs(:,1) = [];
% Remove valores nulos
Customrs = rmmissing(Customrs);
% Atribuindo números aos atributos não numericos
dataset(:,1) = (grp2idx(table2array(Customrs(:,1))));
dataset(:,2:4) = table2array(Customrs(:,2:4));
dataset(:,5) = grp2idx(table2array(Customrs(:,5)));
dataset(:,6:7) = table2array(Customrs(:,6:7));

clear Customrs

%% --------------------  Agrupamento K-Means ------------------------------

% Número de hiperparametros dos K agrupamentos
K = [2 3 4 5];
    
% Critério de parada aplica ao método k-means
criterio_convergencia = 0.00001;

% Número de rodadas 
N = 10;

% Varável que armazena o vetor de silhueta
SWC = zeros(N,length(K));

% Calcula o agrupamento k-means e a silhueta N vezes
for i=1:1:N
    % Cria o modelo k-means
    [clusters,index_clusteres] = k_means(dataset,K,criterio_convergencia);
    % Calcula a métria de silhueta
    SWC(i,:) = metrica_silhueta(dataset,index_clusteres);
end

% Calcula a média das N silhuetas
media_SWC = mean(SWC);
disp("------------- Agrupamento k-means --------------------")
disp("Média de silhueta dos k (k-means) hiperparâmetros na sequência 2, 3, 4 e 5:")
disp(media_SWC)
disp("Melhor hiperparâmetro K de agrupamento: ")
% Escolhe o Melhor hiperparâmetro K
[~,ind_max]= max(media_SWC); 
disp(K(ind_max));

clear SWC ind_max index_clusteres media_SWC

%% --------------------  Agrupamento K-Means com  LDA ---------------------

% Testa o modelo com todos a redução de atributos
for i=1:1:7
    % Aplica a técnica PCA e retorna o dataset reduzido com i atributos
    dataset_reduzido = PCA(dataset,i);

    % Cria o modelo K-means
    [~,index_clusteres_PCA] = k_means(dataset_reduzido,K,criterio_convergencia);
    
    % Calcula a silhueta do modelo com i atributos
    SWC_PCA(i,:) = metrica_silhueta(dataset_reduzido,index_clusteres_PCA);
end

% Pega o melhor número de atributo com base na média dos silhueta
[~,index_melhorNatributo] = max(mean(SWC_PCA,2));
disp("------------- Agrupamento k-means com PCA -------------")
disp("Numeros de atributos selecionados:")
disp(index_melhorNatributo);
disp("Média de silhueta dos k (k-means) hiperparâmetros na sequência 2, 3, 4 e 5:")
disp(SWC_PCA(index_melhorNatributo,:))

clear dataset_reduzido index_melhorNatributo SWC_PCA index_clusteres_PCA i

%% --------------------- Funções implementadas ----------------------------

% Função que calcula os agrupamentos usando o método K_means
function [clusters,index_clusteres] = k_means(dataset,K,criterio_convergencia)
    % Matriz que contem os indices dos grupos 
    index_clusteres = zeros(length(dataset),length(K));
    % tamanho do número de hiperparâmetros testatdos
    tam_ktestes = length(K);
    
    % variável que conterá todos os clusters para cada hiperparâmetro K
    clusters = cell(1, tam_ktestes);
    
    for i=1:1:tam_ktestes
        % Inicializa a semente dos grupos de forma aleatória
        [centroides_iniciais, ~] = inicializacao_centroide(dataset,K(i));
    
        % Agrupa as observações da base de dados conforme os centroides 
        [dataset_clusters, ~] = Clustering(dataset,centroides_iniciais); 
        
        % inicializa os centroides atualizado e o centroide antigo
        news_centroides = centroides_iniciais;
        old_centroides = ones(size(centroides_iniciais)).*1000000;

        % compara a média dos erros entre os centroides atuais e o centroides antigos
        taxa_erro = mean(sum(abs(old_centroides-news_centroides),2));
        
        % Atualiza o centroide até convergir para um erro pequeno entre os centroides atuais e antigos
        while taxa_erro> criterio_convergencia
            old_centroides = news_centroides;
            news_centroides = atualiza_centroide(dataset,dataset_clusters);
            [dataset_clusters,~] = Clustering(dataset,news_centroides); 
           
            taxa_erro = mean(sum(abs(old_centroides-news_centroides),2));
        end
        
        [clusters{1,i}, index_clusteres(:,i)]= Clustering(dataset,news_centroides); 
     end
end

% Função que inicaliza os centroides de forma aleatória
function [centroides, min_max] = inicializacao_centroide(dataset,K)
    tam_dataset = size(dataset);
    % Obtem o minimos e máximos de cada atributo
    min_max = zeros(2,tam_dataset(2));
    for i=1:1:tam_dataset(2)
        min_max(1,i) = min(dataset(:,i));
        min_max(2,i) = max(dataset(:,i));
    end
    
    % Cria os centroides de forma aleatória entre os intervalos minimos e máximos de cada atributo
    centroides = zeros(K,tam_dataset(2));
    for j=1:1:tam_dataset(2)
        centroides(:,j) = min_max(1,j)+ (min_max(2,j)-min_max(1,j)).*rand(K,1);
    end
end

% Função que retorna as distâncias euclidianas entre os centroide e as observações
function distancia = distancia_euclidiana(centroide,dataset)
    tam_dataset = size(dataset); 
    tam_centroides = size(centroide);
    vetor_distancia = zeros(tam_dataset(1),tam_centroides(1));
    
    % Para cada centroide calcula a distância euclidiana
    for j=1:1:tam_centroides(1)
        % Calculo da distância euclidiana
        dist_atributos = (centroide(j,:)-dataset).^2;
        for i=1:1:tam_dataset(1)
            vetor_distancia(i,j) = sum(dist_atributos(i,:));
        end
    end
    % Retorna a distância euclidiana entre as amostras e o centroide
    distancia = (vetor_distancia).^(1/2);
end

% Função que agrupa as observações conforme os centroides
function  [dataset_clusters, index_cluster] = Clustering(dataset,centroides)  
    tam_dataset = size(dataset);
    distancia = distancia_euclidiana(centroides,dataset);
    index_cluster = zeros(tam_dataset(1),1);
    % Analisa qual a menor distância entre os centroides
    for i=1:1:tam_dataset(1)
        [~,ind_minimo] = min(distancia(i,:));
        index_cluster(i,1) = ind_minimo;
    end

    % Divide a base de dados de acordo com os centroides
    unique_centroide = unique(index_cluster);
    for i=1:1:length(unique_centroide)
        dataset_cluster = [];
        ind = 1;
        for j=1:1:tam_dataset(1)
            if index_cluster(j,1) == unique_centroide(i)
                dataset_cluster(ind,:) = dataset(j,:);
                ind = ind + 1;
            end
        end
        dataset_clusters(i,1) = {dataset_cluster};
    end
end

% Função que atualiza os centroides
function news_centroides = atualiza_centroide(dataset,dataset_clusters)
    % Tamanho da base dados
    tam_dataset = size(dataset);
    % Tamanho dos agrupamentos
    tam_clusters = size(dataset_clusters);
    % Matriz que contém os novos centroides 
    news_centroides = zeros(tam_clusters(1),tam_dataset(2));
    
    % atualiza os centroides por meio dos agrupamentos
    for i=1:1:tam_clusters(1)
        dataaux = dataset_clusters{i,1};
        news_centroides(i,:) = mean(dataaux) ;
    end
end


% Função que calcula a média da silhueta de cada K hiperparâmetro
function media_SWC = metrica_silhueta(dataset,index_cluster)
    tam_ktestes = size(index_cluster);
    media_SWC = zeros(1,tam_ktestes(2));

    for i=1:1:tam_ktestes(2)
        media_SWC(:,i) = mean(silhouette(dataset, index_cluster(:,i)));
    end

end

% Função que aplica PCA nos atributos
function dataset_reduzido = PCA(dataset,N_atributos)
    tam_dataset = size(dataset);
    
    % Subtrair a média de cada atributo
    dataset_medianula = dataset - mean(dataset,2);
    
    % Calcula a covariância dos atributos
    covarianca_dataset = cov(dataset_medianula);
    
    % Calcula os autovalores e autovetores da covariância dos atributos
    [autoVetores_dataset, autoValores_dataset] = eig(covarianca_dataset);
    
    % Ordema de forma descrescente as projeções
    [~,index_projecao] = sort(autoValores_dataset*ones(tam_dataset(2),1),'descend');
    projecoes = autoVetores_dataset(:,index_projecao);

    dataset_reduzido = dataset_medianula*projecoes(:,1:N_atributos);
end

