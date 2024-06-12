# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:40:46 2024

@author: User
"""
#Importando librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from utilidades import DimRefuerzo,comprobacion_norma

#%%Función
def Pushover_ML(models_route,NY,NX,LY,LX,FC,W,B_COL,H_COL,B_VIG,H_VIG,CUANTIA_VIG_SUP,CUANTIA_VIG_INF,CUANTIA_COL,range_verify=True):
    #%%Comprobaciones de norma
    A=[NX,NY,LX,LY,FC,W,B_COL,H_COL,B_VIG,H_VIG,CUANTIA_COL,CUANTIA_VIG_SUP,CUANTIA_VIG_INF]#Vector with the user values
    At=["Nx","Ny","Lx","Ly","Fc","w","Bcol","Hcol","Bbeam","Hbeam","ρc","ρb-top","ρb-bot"]
    #Definiendo rango de variables
    Range=[[2,5],[2,5],[4,8],[2.5,4],[17000,35000],[10,30],[0.25,0.5],[0.25,0.5],[0.25,0.5],[0.3,0.65],[0.0112198,0.0278204],[0.0051,0.0124898],[0.0034,0.009088]]#Allowed range
    #Verificacion
    Check=[]
    for i in range(len(A)):
        #Checking the value format (integer or float)
        if(i<=1):
            K=int(A[i])
        else:
            K=float(A[i])
        if range_verify:
            if(K>=Range[i][0])&(K<=Range[i][1]):#Checking the allowed range for each user value
                Check.append(True)
            else:#If user values aren't in the allowed range, the print a warning of the problem
                Check.append(False)
                raise ValueError(f"Value out of range for {At[i]}")
        #Checking normative requirements
        LimH1=np.round(float(0.05*np.round((LX/16)/0.05)),4)#Beam height limit 1
        LimH2=np.round(float(0.05*np.round((LX/12)/0.05)),4)#Beam height limit 2
        LimB1=np.round(float(0.05*np.round((H_VIG/1.4)/0.05)),4)#Beam base limit 1
        LimB2=np.round(float(0.05*np.round((H_VIG/1.2)/0.05)),4)#Beam base limit 2
        OK_P=comprobacion_norma(B_VIG, H_VIG, 0.05, 420000, FC, W, LX, CUANTIA_VIG_SUP, CUANTIA_COL)#Checking SCWB requirement with user values
        if(H_VIG>=LimH1)&(H_VIG<=LimH2)&(B_VIG>=LimB1)&(B_VIG<=LimB2)&(OK_P==1):#Checking if normative requirements are satisfied
            Check.append(True)
        else:#If normative requirements aren't satisfied, the print a warning of the problem
            Check.append(False)
            if(H_VIG<LimH1)|(H_VIG>LimH2):
                raise ValueError("Hbeam don't satisfy norm requirements (Lx)")
            elif(B_VIG<LimB1)|(B_VIG>LimB2):
                raise ValueError("Bbeam don't satisfy norm requirements (Hbeam)")
            elif(OK_P==0):
                raise ValueError("Don't satisfy norm requirements (SCWB)")
    
    #%%Pushover
    recub=0.05#Recubrimiento
    #Columna
    As_C=CUANTIA_COL*B_VIG*H_VIG #Acero necesario
    nB_C,A_barras_C,nLineasAcero_C=DimRefuerzo(B_VIG,As_C,'Columna') #Define el acero comercial y distribución necesaria en columna según cuantía
    C_ColR=(sum(nB_C)*A_barras_C[0])/(B_VIG*B_VIG)
    #Vigas
    As_V=[CUANTIA_VIG_SUP*B_VIG*(H_VIG-recub),CUANTIA_VIG_INF*B_VIG*(H_VIG-recub)]
    n_barras_V,A_barras_V,nLineasAcero_V=DimRefuerzo(B_VIG,As_V,'Viga')  #Define el acero comercial y distribución necesaria en viga según cuantías
    Ct_VigR=(n_barras_V[0]*A_barras_V[0])/(B_VIG*(H_VIG-recub))
    Cb_VigR=(n_barras_V[-1]*A_barras_V[-1])/(B_VIG*(H_VIG-recub))

    #%%Importing the ML models
    #Artificial Neural Networks (ANN)
    red_Plas_D=load_model(f'{models_route}/ANN/ANN_Yielding_D.h5')
    red_Max_D=load_model(f'{models_route}/ANN/ANN_Maximum_D.h5')
    red_Fin_D=load_model(f'{models_route}/ANN/ANN_Failure_D.h5')
    red_Max_Vs=load_model(f'{models_route}/ANN/ANN_Maximum_Vs.h5')
    red_Fin_Vs=load_model(f'{models_route}/ANN/ANN_Failure_Vs.h5')
    #Random Forest (RF)
    regressor_Plas_D = joblib.load(f'{models_route}/RF/RF_Yielding_D.pkl')
    regressor_Max_D = joblib.load(f'{models_route}/RF/RF_Maximum_D.pkl')
    regressor_Fin_D = joblib.load(f'{models_route}/RF/RF_Failure_D.pkl')
    regressor_Max_Vs = joblib.load(f'{models_route}/RF/RF_Maximum_Vs.pkl')
    regressor_Fin_Vs = joblib.load(f'{models_route}/RF/RF_Failure_Vs.pkl')
    
    #%%ML Predictions
    #ANN 
    PRED_RN_Plas_D=red_Plas_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0]
    PRED_RN_Max_D=red_Max_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0]
    PRED_RN_Fin_D=red_Fin_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0] 
    PRED_RN_Max_Vs=red_Max_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0] 
    PRED_RN_Fin_Vs=red_Fin_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0][0] 
    #RF
    PRED_RF_Plas_D=regressor_Plas_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0]
    PRED_RF_Max_D=regressor_Max_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0]
    PRED_RF_Fin_D=regressor_Fin_D.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0] 
    PRED_RF_Max_Vs=regressor_Max_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0] 
    PRED_RF_Fin_Vs=regressor_Fin_Vs.predict([[((NY-Range[0][0])/(Range[0][1]-Range[0][0])),((NX-Range[1][0])/(Range[1][1]-Range[1][0])),((LY-Range[3][0])/(Range[3][1]-Range[3][0])),((LX-Range[2][0])/(Range[2][1]-Range[2][0])),((FC-Range[4][0])/(Range[4][1]-Range[4][0])),((W-Range[5][0])/(Range[5][1]-Range[5][0])),((B_VIG-Range[8][0])/(Range[8][1]-Range[8][0])),((H_VIG-Range[9][0])/(Range[9][1]-Range[9][0])),((Ct_VigR-Range[11][0])/(Range[11][1]-Range[11][0])),((Cb_VigR-Range[12][0])/(Range[12][1]-Range[12][0])),((C_ColR-Range[10][0])/(Range[10][1]-Range[10][0]))]])[0] 
    #Results
    Vs_RF=np.array([0,PRED_RF_Max_Vs*0.7,PRED_RF_Max_Vs,PRED_RF_Fin_Vs])#*Wt
    D_RF=np.array([0,PRED_RF_Plas_D,PRED_RF_Max_D,PRED_RF_Fin_D])#*ht/100
    Vs_RN=np.array([0,PRED_RN_Max_Vs*0.7,PRED_RN_Max_Vs,PRED_RN_Fin_Vs])#*Wt
    D_RN=np.array([0,PRED_RN_Plas_D,PRED_RN_Max_D,PRED_RN_Fin_D])#*ht/100
    
    #%%Grafica
    fig,ax=plt.subplots(dpi=65,figsize=(6,6),facecolor="#EEEEEE")
    plt.yticks(rotation=90)
    plt.grid()
    ax.set_xlabel("Drift/ht")
    ax.set_ylabel("BaseS hear/Wt")
    ax.set_facecolor("#EEEEEE")
    ax.plot(D_RF,Vs_RF, 'r')
    ax.plot(D_RN,Vs_RN, 'b')
    plt.legend(['RF','ANN',"Opensees","Points"])
    
    #%%Resultados
    print("----------------------------------------")
    print("Results:")
    print("  RF:")
    print(f"     Vs: {np.round(Vs_RF,4)}")
    print(f"     δ: {np.round(D_RF,4)}")
    print("  ANN:")
    print(f"     Vs: {np.round(Vs_RN,4)}")
    print(f"     δ: {np.round(D_RN,4)}")
    
#%%Prueba
models_route="D:/Escritorio/ML_PushoverPrediction/GUI/Funcion/ML_Models"
Pushover_ML(models_route,
            NY=5,
            NX=4,
            LY=3.5,
            LX=5,
            FC=24000,
            W=15,
            B_COL=0.3,
            H_COL=0.3,
            B_VIG=0.3,
            H_VIG=0.4,
            CUANTIA_VIG_SUP=0.006,
            CUANTIA_VIG_INF=0.004,
            CUANTIA_COL=0.018,
            range_verify=True)
   



