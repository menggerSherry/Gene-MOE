
################################################################realplace



######对泛癌数据清洗及预处理
rm(list = ls())

# 安装加载需要的R包
library(pacman)
library(readr)
library(matrixStats)
p_load(data.table, tidyverse, magrittr,biomaRt)
project_path="/storage/mxy/SAVAE-Cox/TCGA_pre"
setwd("/storage/mxy/SAVAE-Cox/TCGA_pre/data_new")

pan_data=fread(paste(project_path,"GDC-PANCAN.htseq_fpkm-uq.tsv",sep="/"))

pan_data=pan_data %>%
  filter(str_detect(xena_sample,"^ENSG")) %>%
  mutate(xena_sample = gsub("\\..*", "", xena_sample)) %>%
  rename(Symbol = xena_sample)


pan_data=na.omit(pan_data)


pan_data=pan_data %>%
  rename("Symbol" = 1)



pan_data=pan_data %>%
  column_to_rownames("Symbol")


a=c(colnames(pan_data))

pan_data=pan_data %>%
  mutate(Mean= rowMeans(.[a]), stdev=rowSds(as.matrix(.[a])))

pan_data=pan_data %>%
  filter(stdev>0.4 & Mean>0.7)

pan_data=subset(pan_data,select=-c(Mean,stdev))

fwrite(data.frame(Symbol = rownames(pan_data),pan_data,check.names = F),"GDC-pancancer-clean.tsv",row.names = F, sep = "\t", quote = F)
#########3done!!!!

rm(list = ls())
# 安装加载需要的R包
library(pacman)
library(readr)
p_load(data.table, tidyverse, magrittr,biomaRt)
project_path="/storage/mxy/SAVAE-Cox/TCGA_pre/data_new"
setwd("/storage/mxy/SAVAE-Cox/TCGA_pre/data_new")

pan_data=fread(paste(project_path,"GDC-pancancer-clean.tsv",sep="/"))







############3

##### 对TCGA肿瘤数据预处理
rm(list = ls())

# 安装加载需要的R包
library(pacman)
library(readr)
library(dplyr)
library(matrixStats)
p_load(data.table, tidyverse, magrittr,biomaRt)
project_path="/storage/mxy/SAVAE-Cox/TCGA_pre"
setwd("/storage/mxy/SAVAE-Cox/TCGA_pre/data_new")

#ensembl = biomaRt::useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")
#genes_info = biomaRt::getBM(attributes = c("ensembl_gene_id", "external_gene_name"), filters = "ensembl_gene_id", values = exp_data$ensembl_gene_id, mart = ensembl) # 无对应Symbol的自动删除

################基因过滤
for (data_type in c("TCGA-BLCA","TCGA-BRCA","TCGA-KIRC","TCGA-HNSC","TCGA-LGG", 
  "TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-OV","TCGA-STAD","TCGA-COAD","TCGA-SARC",
  "TCGA-UCEC","TCGA-CESC","TCGA-PRAD","TCGA-SKCM", "TCGA-UCS", "TCGA-THCA","TCGA-THYM", "TCGA-TGCT","TCGA-READ",
  "TCGA-PCPG", "TCGA-PAAD", "TCGA-UVM", "TCGA-MESO", "TCGA-DLBC", "TCGA-KIRP", "TCGA-KICH", "TCGA-GBM", "TCGA-ESCA",
  "TCGA-CHOL", "TCGA-ACC", "TCGA-LAML")){
  #data_type="TCGA-BLCA"
  exp_data_name=paste(data_type,"htseq_fpkm-uq.tsv",sep = ".")
  
  
  exp_data=fread(paste(project_path,exp_data_name,sep = "/"))
  

  
  
  #exp_data=exp_data[rowMeans(exp_data == 0)<0.2,]
  exp_data = exp_data %>%
    filter(str_detect(Ensembl_ID,"^ENSG")) %>%
    mutate(Ensembl_ID = gsub("\\..*", "", Ensembl_ID)) %>% 
    rename(Symbol = Ensembl_ID)
  
  

  
  exp_data=na.omit(exp_data)
  
  exp_data=exp_data %>%
    rename("Symbol" = 1)
  
  exp_data = exp_data %>%
    column_to_rownames("Symbol") 
  
  
  a=c(colnames(exp_data))
  
  exp_data=exp_data %>%
    mutate(Mean= rowMeans(.[a]), stdev=rowSds(as.matrix(.[a])))
  
  exp_data=exp_data %>%
    filter(stdev>0.5 & Mean>0.8)
  
  exp_data=subset(exp_data,select=-c(Mean,stdev))
  
  
  
  
  fwrite(data.frame(Symbol = rownames(exp_data),exp_data,check.names = F),
    paste(data_type,"htseq_fpkm-uq_clean_class.tsv",sep = "."),row.names = F, sep = "\t", quote = F)
  
}

##########################取重合的基因

rm(list = ls())

# 安装加载需要的R包
library(pacman)
library(readr)
p_load(data.table, tidyverse, magrittr)
project_path="/storage/mxy/SAVAE-Cox/TCGA_pre/data_new"
setwd("/storage/mxy/SAVAE-Cox/TCGA_pre/data_new")

pancancer_data=fread(paste(project_path,"GDC-pancancer-clean.tsv",sep="/"))
blca_data=fread(paste(project_path,"TCGA-BLCA.htseq_fpkm-uq_clean_class.tsv",sep="/"))
brca_data=fread(paste(project_path,"TCGA-BRCA.htseq_fpkm-uq_clean_class.tsv",sep ="/"))
kirc_data=fread(paste(project_path,"TCGA-KIRC.htseq_fpkm-uq_clean_class.tsv",sep = "/"))
hnsc_data=fread(paste(project_path,"TCGA-HNSC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
lgg_data=fread(paste(project_path,"TCGA-LGG.htseq_fpkm-uq_clean_class.tsv",sep="/"))
lihc_data=fread(paste(project_path,"TCGA-LIHC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
luad_data=fread(paste(project_path,"TCGA-LUAD.htseq_fpkm-uq_clean_class.tsv",sep="/"))
lusc_data=fread(paste(project_path,"TCGA-LUSC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
ov_data=fread(paste(project_path,"TCGA-OV.htseq_fpkm-uq_clean_class.tsv",sep="/"))
coad_data=fread(paste(project_path,"TCGA-COAD.htseq_fpkm-uq_clean_class.tsv",sep="/"))
sarc_data=fread(paste(project_path,"TCGA-SARC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
ucec_data=fread(paste(project_path,"TCGA-UCEC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
stad_data=fread(paste(project_path,"TCGA-STAD.htseq_fpkm-uq_clean_class.tsv",sep="/"))
cesc_data=fread(paste(project_path,"TCGA-CESC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
prad_data=fread(paste(project_path,"TCGA-PRAD.htseq_fpkm-uq_clean_class.tsv",sep="/"))
skcm_data=fread(paste(project_path,"TCGA-SKCM.htseq_fpkm-uq_clean_class.tsv",sep="/"))
ucs_data=fread(paste(project_path,"TCGA-UCS.htseq_fpkm-uq_clean_class.tsv",sep="/"))
thca_data=fread(paste(project_path,"TCGA-THCA.htseq_fpkm-uq_clean_class.tsv",sep ="/"))
thym_data=fread(paste(project_path,"TCGA-THYM.htseq_fpkm-uq_clean_class.tsv",sep = "/"))
tgct_data=fread(paste(project_path,"TCGA-TGCT.htseq_fpkm-uq_clean_class.tsv",sep="/"))
read_data=fread(paste(project_path,"TCGA-READ.htseq_fpkm-uq_clean_class.tsv",sep="/"))
pcpg_data=fread(paste(project_path,"TCGA-PCPG.htseq_fpkm-uq_clean_class.tsv",sep="/"))
paad_data=fread(paste(project_path,"TCGA-PAAD.htseq_fpkm-uq_clean_class.tsv",sep="/"))
uvm_data=fread(paste(project_path,"TCGA-UVM.htseq_fpkm-uq_clean_class.tsv",sep="/"))
meso_data=fread(paste(project_path,"TCGA-MESO.htseq_fpkm-uq_clean_class.tsv",sep="/"))
dlbc_data=fread(paste(project_path,"TCGA-DLBC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
kirp_data=fread(paste(project_path,"TCGA-KIRP.htseq_fpkm-uq_clean_class.tsv",sep="/"))
kich_data=fread(paste(project_path,"TCGA-KICH.htseq_fpkm-uq_clean_class.tsv",sep="/"))
gbm_data=fread(paste(project_path,"TCGA-GBM.htseq_fpkm-uq_clean_class.tsv",sep="/"))
esca_data=fread(paste(project_path,"TCGA-ESCA.htseq_fpkm-uq_clean_class.tsv",sep="/"))
chol_data=fread(paste(project_path,"TCGA-CHOL.htseq_fpkm-uq_clean_class.tsv",sep="/"))
acc_data=fread(paste(project_path,"TCGA-ACC.htseq_fpkm-uq_clean_class.tsv",sep="/"))
laml_data=fread(paste(project_path,"TCGA-LAML.htseq_fpkm-uq_clean_class.tsv",sep="/"))
# common=intersect(as.character(as.matrix(blca_data$Symbol)),as.character(as.matrix(brca_data$Symbol)),
#   as.character(as.matrix(cesc_data$Symbol)),as.character(as.matrix(kirc_data$Symbol)),as.character(as.matrix(pancancer_data$Symbol)))

common=Reduce(intersect,list(v1=as.character(as.matrix(blca_data$Symbol)),
  v2=as.character(as.matrix(brca_data$Symbol)),
  v3=as.character(as.matrix(kirc_data$Symbol)),
  v4=as.character(as.matrix(hnsc_data$Symbol)),
  v5=as.character(as.matrix(lgg_data$Symbol)),
  v6=as.character(as.matrix(lihc_data$Symbol)),
  v7=as.character(as.matrix(luad_data$Symbol)),
  v8=as.character(as.matrix(lusc_data$Symbol)),
  v9=as.character(as.matrix(ov_data$Symbol)),
  v10=as.character(as.matrix(stad_data$Symbol)),
  v11=as.character(as.matrix(pancancer_data$Symbol)),
  v12=as.character(as.matrix(coad_data$Symbol)),
  v13=as.character(as.matrix(sarc_data$Symbol)),
  v14=as.character(as.matrix(ucec_data$Symbol)),
  v15=as.character(as.matrix(cesc_data$Symbol)),
  v16=as.character(as.matrix(prad_data$Symbol)),
  v17=as.character(as.matrix(skcm_data$Symbol)),
  v18=as.character(as.matrix(ucs_data$Symbol)),
  v19=as.character(as.matrix(thca_data$Symbol)),
  v20=as.character(as.matrix(thym_data$Symbol)),
  v21=as.character(as.matrix(tgct_data$Symbol)),
  v22=as.character(as.matrix(read_data$Symbol)),
  v23=as.character(as.matrix(pcpg_data$Symbol)),
  v24=as.character(as.matrix(paad_data$Symbol)),
  v25=as.character(as.matrix(uvm_data$Symbol)),
  v26=as.character(as.matrix(meso_data$Symbol)),
  v27=as.character(as.matrix(dlbc_data$Symbol)),
  v28=as.character(as.matrix(kirp_data$Symbol)),
  v29=as.character(as.matrix(kich_data$Symbol)),
  v30=as.character(as.matrix(gbm_data$Symbol)),
  v31=as.character(as.matrix(esca_data$Symbol)),
  v32=as.character(as.matrix(chol_data$Symbol)),
  v33=as.character(as.matrix(acc_data$Symbol)),
  v34=as.character(as.matrix(laml_data$Symbol))
))

blca_data %<>% filter(Symbol %in% common)
brca_data %<>% filter(Symbol %in% common)
kirc_data %<>% filter(Symbol %in% common)
hnsc_data %<>% filter(Symbol %in% common)
lgg_data %<>% filter(Symbol %in% common)
lihc_data %<>% filter(Symbol %in% common)
luad_data %<>% filter(Symbol %in% common)
lusc_data %<>% filter(Symbol %in% common)
ov_data %<>% filter(Symbol %in% common)
stad_data %<>% filter(Symbol %in% common)
coad_data %<>% filter(Symbol %in% common)
sarc_data %<>% filter(Symbol %in% common)
ucec_data %<>% filter(Symbol %in% common)
cesc_data %<>% filter(Symbol %in% common)
prad_data %<>% filter(Symbol %in% common)
skcm_data %<>% filter(Symbol %in% common)
pancancer_data %<>% filter(Symbol %in% common)
ucs_data %<>% filter(Symbol %in% common)
thca_data %<>% filter(Symbol %in% common)
thym_data %<>% filter(Symbol %in% common)
tgct_data %<>% filter(Symbol %in% common)
read_data %<>% filter(Symbol %in% common)
pcpg_data %<>% filter(Symbol %in% common)
paad_data %<>% filter(Symbol %in% common)
uvm_data %<>% filter(Symbol %in% common)
meso_data %<>% filter(Symbol %in% common)
dlbc_data %<>% filter(Symbol %in% common)
kirp_data %<>% filter(Symbol %in% common)
kich_data %<>% filter(Symbol %in% common)
gbm_data %<>% filter(Symbol %in% common)
esca_data %<>% filter(Symbol %in% common)
chol_data %<>% filter(Symbol %in% common)
acc_data %<>% filter(Symbol %in% common)
laml_data %<>% filter(Symbol %in% common)

fwrite(blca_data,"TCGA-BLCA.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(brca_data,"TCGA-BRCA.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(kirc_data,"TCGA-KIRC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(hnsc_data,"TCGA-HNSC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(lgg_data,"TCGA-LGG.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(lihc_data,"TCGA-LIHC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(luad_data,"TCGA-LUAD.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(lusc_data,"TCGA-LUSC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(ov_data,"TCGA-OV.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(stad_data,"TCGA-STAD.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(coad_data,"TCGA-COAD.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(sarc_data,"TCGA-SARC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(ucec_data,"TCGA-UCEC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(cesc_data,"TCGA-CESC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(prad_data,"TCGA-PRAD.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(skcm_data,"TCGA-SKCM.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(pancancer_data,"GDC_PANCANCER.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(ucs_data,"TCGA-UCS.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(thca_data,"TCGA-THCA.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(thym_data,"TCGA-THYM.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(tgct_data,"TCGA-TGCT.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(read_data,"TCGA-READ.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(pcpg_data,"TCGA-PCPG.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(paad_data,"TCGA-PAAD.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(uvm_data,"TCGA-UVM.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(meso_data,"TCGA-MESO.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(dlbc_data,"TCGA-DLBC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(kirp_data,"TCGA-KIRP.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(kich_data,"TCGA-KICH.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(gbm_data,"TCGA-GBM.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(esca_data,"TCGA-ESCA.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(chol_data,"TCGA-CHOL.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(acc_data,"TCGA-ACC.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)
fwrite(laml_data,"TCGA-LAML.htseq_fpkm-uq_final.tsv", row.names = F, sep = "\t", quote = F)











##### 匹配诊断数据
rm(list = ls())

# 安装加载需要的R包
library(pacman)
library(readr)
library(dplyr)
library(matrixStats)
p_load(data.table, tidyverse, magrittr,biomaRt)
project_path="/storage/mxy/SAVAE-Cox/TCGA_pre/data_new"
project_path2 = "/storage/mxy/SAVAE-Cox/TCGA_pre"
setwd("/storage/mxy/SAVAE-Cox/TCGA_pre/data_new")

#ensembl = biomaRt::useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")
#genes_info = biomaRt::getBM(attributes = c("ensembl_gene_id", "external_gene_name"), filters = "ensembl_gene_id", values = exp_data$ensembl_gene_id, mart = ensembl) # 无对应Symbol的自动删除


for (data_type in c("TCGA-BLCA","TCGA-BRCA","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-OV","TCGA-STAD","TCGA-COAD","TCGA-SARC","TCGA-UCEC","TCGA-CESC","TCGA-PRAD","TCGA-SKCM")){
  #data_type="TCGA-BLCA"
  exp_data_name=paste(data_type,"htseq_fpkm-uq_final.tsv",sep = ".")
  sur_data_name=paste(data_type,"survival.tsv",sep = ".")
  
  exp_data=fread(paste(project_path,exp_data_name,sep = "/"))
  sur_data=fread(paste(project_path2,sur_data_name,sep="/"))
  
  
  #exp_data=exp_data[rowMeans(exp_data == 0)<0.2,]
  
  
  #ensembl = biomaRt::useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")
  #genes_info = biomaRt::getBM(attributes = c("ensembl_gene_id", "external_gene_name"), filters = "ensembl_gene_id", values = exp_data$ensembl_gene_id, mart = ensembl) # 无对应Symbol的自动删除
  
  #exp_data=inner_join(exp_data, genes_info, by = "ensembl_gene_id", keep = F) %>%
  # mutate_if(is.character, list(~na_if(.,""))) 
  #exp_data=na.omit(exp_data)
  
  #exp_data=exp_data %>%
  # unite("name_id", external_gene_name, ensembl_gene_id, sep = "|", remove = T) %>%
  #rename("Symbol" = 1)
  
  exp_data = exp_data %>%
    column_to_rownames("Symbol") %>%
    dplyr::select(ends_with("-01A"))
  
  
  sur_data = sur_data %>%
    filter(OS.time>0 & !is.na(OS)) %>%
    droplevels
  
  #common=intersect(as.character(as.matrix(sur_data$sample)),colnames(exp_data))
  common=Reduce(intersect,list(v1=as.character(as.matrix(sur_data$sample)),
    v2=colnames(exp_data)
  ))
  
  sur_data %<>% filter(sample %in% common)
  exp_data=exp_data[,common]
  
  fwrite(sur_data,paste(data_type,"survival_clean.tsv",sep = "."),row.names = F, sep = "\t", quote = F)
  fwrite(data.frame(Symbol = rownames(exp_data),exp_data,check.names = F),
    paste(data_type,"htseq_fpkm-uq_finalsurviva.tsv",sep = "."),row.names = F, sep = "\t", quote = F)
  
}


#################用于癌症分类: htseq_fpkm-uq_final  用于生存分析： htseq_fpkm-uq_finalsurviva
