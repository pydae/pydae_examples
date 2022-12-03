PSCAD 4.2.0

Settings
 {
 Id = "912544816.1668442251"
 Author = "jayas.jmmau"
 Desc = "GENERATOR WITH EXCITER"
 Arch = "windows"
 Options = 0
 Build = 18
 Warn = 1
 Check = 15
 Libs = ""
 Source = ""
 RunInfo = 
  {
  Fin = 20
  Step = 5e-005
  Plot = 0.005
  Chat = 0.001
  Brch = 0.0005
  Lat = 100
  Options = 0
  Advanced = 511
  Debug = 0
  StartFile = ""
  OFile = "Untitled.out"
  SFile = "Untitled.snp"
  SnapTime = 0.3
  Mruns = 10
  Mrunfile = 0
  StartType = 0
  PlotType = 0
  SnapType = 0
  MrunType = "mrun"
  }

 }

Definitions
 {
 Module("Main")
  {
  Desc = " EXAMPLE 2"
  FileDate = 0
  Nodes = 
   {
   }

  Graphics = 
   {
   Rectangle(-18,-18,18,18)
   }


  Page(A/A4,Portrait,16,[640,323],100)
   {
   -Wire-([288,432],0,0,-1)
    {
    Vertex="0,0;144,0"
    }
   0.pgb([144,1152],0,106873488,160)
    {
    Name = "TERMINAL VOLTAGE"
    Group = "Terminal Voltage"
    Display = "0"
    Scale = "1.0"
    Units = "kV"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "20"
    }
   0.datalabel([180,504],4,0,-1)
    {
    Name = "W"
    }
   0.sync_machine([216,432],0,0,250)
    {
    Name = "HG"
    Nqaxw = "1"
    Cnfg = "0"
    MM = "1"
    CfRa = "1"
    MSat = "0"
    icTyp = "1"
    Iscl = "1"
    View = "1"
    itrfa = "0"
    izro = "0"
    Icsp = "0"
    Icmp = "0"
    Immd = "0"
    Ifmlt = "0"
    Term = "3"
    Ts = "0.02 [s]"
    Iexc = "1"
    Igov = "1"
    Ospd = "0"
    machsw = "S2M"
    Enab = "ENAB"
    npadjs = "0"
    pset = "0.0 [MW]"
    nftrsw = "0"
    hmult = "1.0"
    sdmftr = "1.0"
    sdmspd = "1.0"
    npadjm = "0"
    fldmlt = "1.0"
    nfldsw = "0"
    Vbase = "7.967 [kV]"
    Ibase = "5.02 [kA]"
    OMO = "376.991118 [rad/s]"
    H = "5 [s]"
    D = "0.04 [pu]"
    RNeut = "1.0E5 [pu]"
    XNeut = "0 [pu]"
    Ri = "300.0 [pu]"
    NOM = "1.0"
    Rs1 = "0.0025 [pu]"
    XS1 = "0.14 [pu]"
    XMD0 = "1.66 [pu]"
    R2D = "0.00043 [pu]"
    X2D = "0.2004 [pu]"
    R3D = "0.0051 [pu]"
    X3D = "0.0437 [pu]"
    X230 = "0.0 [pu]"
    XMQ = "0.91 [pu]"
    R2Q = "0.00842 [pu]"
    X2Q = "0.106 [pu]"
    R3Q = "8.1942E-03 [pu]"
    X3Q = "9.4199E-02 [pu]"
    X231 = "0.0 [pu]"
    Ra = "0.0051716 [pu]"
    Ta = "0.278 [s]"
    Xp = "0.163 [pu]"
    Xd = "1.014 [pu]"
    Xd' = "0.314 [pu]"
    Tdo' = "6.55 [s]"
    Xd'' = "0.280 [pu]"
    Tdo'' = "0.039 [s]"
    Gfld = "1.0E+2 [pu]"
    Xkf = "0.0 [pu]"
    Xq = "0.770 [pu]"
    Xq' = "0.228 [pu]"
    Tqo' = "0.85 [s]"
    Xq'' = "0.375 [pu]"
    Tqo'' = "0.071 [s]"
    AGFC = "1.0"
    X1 = "0.0"
    Y1 = "0.0 [pu]"
    X2 = "0.5"
    Y2 = "0.5 [pu]"
    X3 = "0.8"
    Y3 = "0.79 [pu]"
    X4 = "1.0"
    Y4 = "0.947 [pu]"
    X5 = "1.2"
    Y5 = "1.076 [pu]"
    X6 = "1.5"
    Y6 = "1.2 [pu]"
    X7 = "1.8"
    Y7 = "1.26 [pu]"
    X8 = "2.2"
    Y8 = "1.32 [pu]"
    X9 = "3.2"
    Y9 = "1.53 [pu]"
    X10 = "4.2"
    Y10 = "1.74 [pu]"
    VT = "1.0[pu]"
    Pheta = "0 [rad]"
    Trmpv = "0.1 [s]"
    Sysfl = "100.0 [pu]"
    Ptcon = "0.2 [s]"
    P0 = "60.0 [MW]"
    Q0 = "0.0 [MVAR]"
    Theti = "3.141592 [rad]"
    Idi = "0.0 [pu]"
    Iqi = "0.0 [pu]"
    Ifi = "0.0 [pu]"
    Spdi = "1.0 [pu]"
    POut = "POUT"
    QOut = "QOUT"
    Vneut = ""
    Cneut = ""
    Lang = "Lang"
    Theta = "Theta"
    Wang = "Wang"
    Tesmt = ""
    PQscl = "0"
    InExc = "InitEx"
    InGov = "InitGv"
    Mon1 = "1"
    Chn1 = ""
    Mon2 = "1"
    Chn2 = ""
    Mon3 = "1"
    Chn3 = ""
    Mon4 = "1"
    Chn4 = ""
    Mon5 = "1"
    Chn5 = ""
    Mon6 = "1"
    Chn6 = ""
    }
   0.datalabel([180,360],1,0,-1)
    {
    Name = "EF"
    }
   -Wire-([432,468],0,0,-1)
    {
    Vertex="0,0;0,-36"
    }
   0.breaker3([432,504],3,0,380)
    {
    Ctrl = "0"
    OPCUR = "0"
    ENAB = "0"
    CLVL = "0.0 [kA]"
    View = "1"
    ViewB = "0"
    DisPQ = "0"
    NAME = "DIST"
    NAMEA = "BRKA"
    NAMEB = "BRKB"
    NAMEC = "BRKC"
    ROFF = "1.0E6 [ohm]"
    RON = "0.01 [ohm]"
    PRER = "0.5 [ohm]"
    TDA = "0.0 [s]"
    TDB = "0.0 [s]"
    TDC = "0.0 [s]"
    TDRA = "0.05 [s]"
    TDRB = "0.05 [s]"
    TDRC = "0.05 [s]"
    PostIns = "0"
    TDBOA = "0.005 [s]"
    IBRA = ""
    IBRB = ""
    IBRC = ""
    IBR0 = ""
    SBRA = ""
    SBRB = ""
    SBRC = ""
    BP = ""
    BQ = ""
    BOpen1 = "0"
    BOpen2 = "0"
    BOpen3 = "0"
    P = "0 [MW]"
    Q = "0 [MVAR]"
    }
   -Wire-([432,648],0,0,-1)
    {
    Vertex="0,0;0,-36"
    }
   0.ground([432,648],1,0,-1)
    {
    }
   0.inductor([432,612],3,0,-1)
    {
    L = "0.00842 [H]"
    }
   -Wire-([432,576],0,0,-1)
    {
    Vertex="0,0;0,-36"
    }
   0.annotation([432,684],0,0,-1)
    {
    AL1 = "2 PU"
    AL2 = "INDUCTANCE"
    }
   0.annotation([216,666],2,0,-1)
    {
    AL1 = "HYDRO   GENERATOR"
    AL2 = "120 MVA 13.8 kV"
    }
   0.datalabel([144,468],4,0,-1)
    {
    Name = "TM"
    }
   0.datalabel([324,288],4,0,-1)
    {
    Name = "DIST"
    }
   0.select([324,252],4,0,200)
    {
    A = "1"
    DPath = "1"
    COM = "Selector"
    }
   0.datalabel([216,360],3,0,-1)
    {
    Name = "IF"
    }
   -Sticky-([72,18],0)
    {
    Name = "Untitled"
    Font = 2
    Bounds = 72,18,1152,54
    Alignment = 1
    Style = 0
    Arrow = 0
    Color = 0,15792890
    Text = "SYNCHRONOUS MACHINE WITH EXCITER"
    }
   0.sandhdefn([252,540],4,0,260)
    {
    Iand = "1"
    }
   -Wire-([216,504],0,0,-1)
    {
    Vertex="0,0;0,36"
    }
   0.datalabel([252,576],0,0,-1)
    {
    Name = "ENAB"
    }
   0.excac([216,324],0,0,240)
    {
    TYPE = "1"
    INIT = "InitEx"
    OVR = "1"
    RC = "0.0 [pu]"
    XC = "0.0 [pu]"
    TR = "0.0 [s]"
    STAB = "1"
    TC_1 = "0.0 [s]"
    TB_1 = "0.0 [s]"
    KA_1 = "200.0 [pu]"
    TA_1 = "0.02 [s]"
    VAMX_1 = "14.5 [pu]"
    VAMN_1 = "-14.5 [pu]"
    UEL_1 = "0"
    OEL_1 = "0"
    VRMX_1 = "6.03 [pu]"
    VRMN_1 = "-5.43 [pu]"
    KF_1 = "0.03 [pu]"
    TF_1 = "1.0 [s]"
    TE_1 = "0.80 [s]"
    KE_1 = "1.00 [pu]"
    KC_1 = "0.20 [pu]"
    KD_1 = "0.38 [pu]"
    SE1_1 = "0.10 [pu]"
    VE1_1 = "4.18 [pu]"
    SE2_1 = "0.03 [pu]"
    VE2_1 = "3.14 [pu]"
    VUEL_1 = "-1.0E10 [pu]"
    VOEL_1 = "1.0E10 [pu]"
    TC_2 = "0.0 [s]"
    TB_2 = "0.0 [s]"
    KA_2 = "400.0 [pu]"
    TA_2 = "0.01 [s]"
    VAMX_2 = "8.0 [pu]"
    VAMN_2 = "-8.0 [pu]"
    KB_2 = "25.0 [pu]"
    UEL_2 = "0"
    OEL_2 = "0"
    VRMX_2 = "105. [pu]"
    VRMN_2 = "-95. [pu]"
    KF_2 = "0.03 [pu]"
    TF_2 = "1.0 [s]"
    KH_2 = "1.0 [pu]"
    TE_2 = "0.60 [s]"
    VFMX_2 = "4.4 [pu]"
    KE_2 = "1.00 [pu]"
    KC_2 = "0.28 [pu]"
    KD_2 = "0.35 [pu]"
    SE1_2 = "0.037 [pu]"
    VE1_2 = "4.4 [pu]"
    SE2_2 = "0.012 [pu]"
    VE2_2 = "3.3 [pu]"
    VUEL_2 = "-1.0E10 [pu]"
    VOEL_2 = "1.0E10 [pu]"
    TC_3 = "0.0 [s]"
    TB_3 = "0.0 [s]"
    UEL_3 = "0"
    KA_3 = "45.62 [pu]"
    TA_3 = "0.013 [s]"
    VAMX_3 = "1.0 [pu]"
    VAMN_3 = "-0.95 [pu]"
    KR_3 = "3.77 [pu]"
    VLV_3 = "0.790 [pu]"
    KLV_3 = "0.194 [pu]"
    KF_3 = "0.143 [pu]"
    KN_3 = "0.05 [pu]"
    EFDN_3 = "2.36 [pu]"
    TF_3 = "1.0 [s]"
    TE_3 = "1.17 [s]"
    VFMX_3 = "16.0 [pu]"
    KE_3 = "1.00 [pu]"
    KC_3 = "0.104 [pu]"
    KD_3 = "0.499 [pu]"
    SE1_3 = "1.143 [pu]"
    VE1_3 = "6.24 [pu]"
    SE2_3 = "0.10 [pu]"
    VE2_3 = "4.68 [pu]"
    VUEL_3 = "-1.0E10 [pu]"
    VIMX_4 = "10.0 [pu]"
    VIMN_4 = "-10.0 [pu]"
    TC_4 = "1.0 [s]"
    TB_4 = "10.0 [s]"
    UEL_4 = "0"
    KA_4 = "200.0 [pu]"
    TA_4 = "0.015 [s]"
    VRMX_4 = "5.64 [pu]"
    VRMN_4 = "-4.53 [pu]"
    KC_4 = "0.0 [pu]"
    VUEL_4 = "-1.0E10 [pu]"
    KA_5 = "400.0 [pu]"
    TA_5 = "0.02 [s]"
    VRMX_5 = "7.3 [pu]"
    VRMN_5 = "-7.3 [pu]"
    TE_5 = "0.80 [s]"
    KE_5 = "1.00 [pu]"
    SE1_5 = "0.86 [pu]"
    EF1_5 = "5.60 [pu]"
    SE2_5 = "0.50 [pu]"
    EF2_5 = "4.20 [pu]"
    KF_5 = "0.03 [pu]"
    TF1_5 = "1.0 [s]"
    TF2_5 = "0.0 [s]"
    TF3_5 = "0.0 [s]"
    UEL_6 = "0"
    KA_6 = "53.6 [pu]"
    TK_6 = "0.18 [s]"
    TA_6 = "0.086 [s]"
    TC_6 = "3.0 [s]"
    TB_6 = "9.0 [s]"
    VAMX_6 = "75. [pu]"
    VAMN_6 = "-75. [pu]"
    VRMX_6 = "44. [pu]"
    VRMN_6 = "-36. [pu]"
    TE_6 = "1.0 [s]"
    KE_6 = "1.6 [pu]"
    KC_6 = "0.173 [pu]"
    KD_6 = "1.91 [pu]"
    SE1_6 = "0.214 [pu]"
    VE1_6 = "7.4 [pu]"
    SE2_6 = "0.044 [pu]"
    VE2_6 = "5.55 [pu]"
    VFLM_6 = "19.0 [pu]"
    KH_6 = "92.0 [pu]"
    VHMX_6 = "75.0 [pu]"
    TH_6 = "0.08 [s]"
    TJ_6 = "0.02 [s]"
    VUEL_6 = "0.00 [pu]"
    UEL_7 = "0"
    KRP_7 = "12.77 [pu]"
    KRI_7 = "20.0 [pu]"
    VRMX_7 = "5.0 [pu]"
    VRMN_7 = "-5.0 [pu]"
    KAP_7 = "20. [pu]"
    KAI_7 = "1.0 [pu]"
    VAMX_7 = "1.0 [pu]"
    VAMN_7 = "-1.0 [pu]"
    KP_7 = "6.41 [pu]"
    VLV_7 = "0.79 [pu]"
    KL_7 = "26.2 [pu]"
    VFMX_7 = "6.1 [pu]"
    SE1_7 = "1.195 [pu]"
    VE1_7 = "4.025 [pu]"
    SE2_7 = "0.097 [pu]"
    VE2_7 = "3.02 [pu]"
    TE_7 = "1.945 [s]"
    KE_7 = "1.0 [pu]"
    KD_7 = "0.567 [pu]"
    KC_7 = "0.172 [pu]"
    KF_7 = "1.0 [pu]"
    VUEL_7 = "0.0 [pu]"
    KP_8 = "17.0 [pu]"
    KI_8 = "13.0 [pu]"
    KD_8 = "6.0 [pu]"
    TD_8 = "0.03 [s]"
    KA_8 = "1.0 [pu]"
    TA_8 = "0.0 [s]"
    VRMX_8 = "10.0 [pu]"
    VRMN_8 = "0.0 [pu]"
    TE_8 = "1.0 [s]"
    KE_8 = "1.0 [pu]"
    SE1_8 = "1.5 [pu]"
    EF1_8 = "4.5 [pu]"
    SE2_8 = "1.36 [pu]"
    EF2_8 = "3.38 [pu]"
    }
   -Wire-([144,324],0,0,-1)
    {
    Vertex="0,0;0,36"
    }
   -Wire-([252,324],0,0,-1)
    {
    Vertex="0,0;0,36"
    }
   0.sandhdefn([180,180],0,0,270)
    {
    Iand = "1"
    }
   -Wire-([144,180],0,0,-1)
    {
    Vertex="0,0;0,72;36,72"
    }
   0.datalabel([180,216],0,0,-1)
    {
    Name = "S2M"
    }
   0.sumjct([252,252],2,0,280)
    {
    DPath = "1"
    A = "0"
    B = "0"
    C = "0"
    D = "1"
    E = "0"
    F = "1"
    G = "0"
    }
   0.const([396,216],4,0,40)
    {
    Name = ""
    Value = "0.0"
    }
   0.const([396,252],4,0,50)
    {
    Name = ""
    Value = "0.01"
    }
   0.compare([432,1170],0,0,180)
    {
    X = "1.0"
    OL = "1"
    OH = "0"
    INTR = "0"
    }
   0.datalabel([504,1170],3,0,-1)
    {
    Name = "DIST"
    }
   0.time-sig([360,1170],0,0,170)
    {
    }
   0.datalabel([504,1044],3,0,-1)
    {
    Name = "ENAB"
    }
   0.compare([432,1044],0,0,140)
    {
    X = "0.5"
    OL = "0"
    OH = "1"
    INTR = "0"
    }
   0.time-sig([360,1044],0,0,130)
    {
    }
   0.annotation([432,1080],0,0,-1)
    {
    AL1 = "RELEASE MACHINE"
    AL2 = "@ 0.5 SEC."
    }
   0.annotation([432,1206],0,0,-1)
    {
    AL1 = "DISTURBANCE"
    AL2 = "@ 1.0 SEC."
    }
   0.pgb([252,990],0,106886544,320)
    {
    Name = "FIELD VOLTAGE"
    Group = "Field Voltage"
    Display = "0"
    Scale = "1.0"
    Units = "pu"
    mrun = "0"
    Pol = "0"
    Min = "-5"
    Max = "6"
    }
   0.pgb([144,918],0,106882464,370)
    {
    Name = "REAL POWER"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "120"
    Units = "MW"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "120"
    }
   0.pgb([144,954],0,106890624,350)
    {
    Name = "REACTIVE POWER"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "120"
    Units = "MVAR"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "120"
    }
   0.pgb([144,990],0,106882872,330)
    {
    Name = "FIELD CURRENT"
    Group = "Field Current"
    Display = "0"
    Scale = "1"
    Units = "pu"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "2"
    }
   -Wire-([108,918],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   -Wire-([108,954],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   -Wire-([108,990],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.datalabel([108,918],0,0,-1)
    {
    Name = "POUT"
    }
   0.datalabel([108,954],0,0,-1)
    {
    Name = "QOUT"
    }
   0.datalabel([108,990],0,0,-1)
    {
    Name = "IF"
    }
   0.pgb([252,918],0,106883280,360)
    {
    Name = "MECHANICAL TORQUE"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "1.0"
    Units = "pu"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "1"
    }
   -Wire-([216,918],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.datalabel([216,954],0,0,-1)
    {
    Name = "W"
    }
   0.pgb([252,954],0,106880832,340)
    {
    Name = "OMEGA"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "1.0"
    Units = "rad/s"
    mrun = "0"
    Pol = "0"
    Min = "376"
    Max = "382"
    }
   -Wire-([216,954],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   -Wire-([216,990],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.datalabel([216,990],0,0,-1)
    {
    Name = "EF"
    }
   0.datalabel([216,918],0,0,-1)
    {
    Name = "TM"
    }
   0.datalabel([108,1044],0,0,-1)
    {
    Name = "Lang"
    }
   0.pgb([144,1044],0,106883688,310)
    {
    Name = "Load Angle"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "1"
    Units = "MW"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "120"
    }
   0.datalabel([108,1080],0,0,-1)
    {
    Name = "Theta"
    }
   0.pgb([144,1080],0,106884504,300)
    {
    Name = "Theta"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "1"
    Units = "MW"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "120"
    }
   0.datalabel([108,1116],0,0,-1)
    {
    Name = "Wang"
    }
   0.pgb([144,1116],0,106885320,290)
    {
    Name = "Internal Phase Angle"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "1"
    Units = "MW"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "120"
    }
   0.unity([504,1044],0,0,150)
    {
    IType = "2"
    OType = "1"
    }
   0.datalabel([504,918],3,0,-1)
    {
    Name = "S2M"
    }
   0.compare([432,918],0,0,110)
    {
    X = "0.3"
    OL = "0"
    OH = "1"
    INTR = "0"
    }
   0.time-sig([360,918],0,0,100)
    {
    }
   0.annotation([432,954],0,0,-1)
    {
    AL1 = "SOURCE -> MACHINE"
    AL2 = "@ 0.3 SEC."
    }
   0.unity([504,918],0,0,120)
    {
    IType = "2"
    OType = "1"
    }
   -Wire-([108,1044],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   -Wire-([108,1080],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   -Wire-([108,1116],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.multimeter([270,432],0,0,90)
    {
    MeasI = "0"
    MeasV = "0"
    MeasP = "0"
    MeasQ = "0"
    RMS = "1"
    MeasPh = "0"
    S = "1.0 [MVA]"
    BaseV = "1.0 [kV]"
    TS = "0.02 [s]"
    Freq = "60.0 [Hz]"
    Dis = "0"
    CurI = ""
    VolI = ""
    P = ""
    Q = ""
    Vrms = "Vrms"
    Ph = ""
    hide1 = "0"
    hide2 = "0"
    Pd = ""
    Qd = ""
    Vd = ""
    }
   -Wire-([288,540],0,0,-1)
    {
    Vertex="0,0;0,-36;-36,-36"
    }
   -Wire-([252,216],0,0,-1)
    {
    Vertex="0,0;0,-36;-36,-36"
    }
   0.datalabel([108,1152],0,0,-1)
    {
    Name = "Vrms"
    }
   -Wire-([108,1152],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.unity([504,1170],0,0,190)
    {
    IType = "2"
    OType = "1"
    }
   0.source_3([540,324],7,0,390)
    {
    Name = "Source 1"
    Type = "4"
    Grnd = "1"
    View = "1"
    Spec = "0"
    VCtrl = "1"
    FCtrl = "1"
    Vm = "230.0 [kV]"
    Tc = "0.05 [s]"
    f = "60.0 [Hz]"
    Ph = "0.0 [deg]"
    Vbase = "230.0 [kV]"
    Sbase = "100.0 [MVA]"
    Vpu = "1.0 [pu]"
    PhT = "0.0 [deg]"
    Pinit = "0.0 [pu]"
    Qinit = "0.0 [pu]"
    R = "1e-6[ohm]"
    Rs = "1.0 [ohm]"
    Rp = "1.0 [ohm]"
    Lp = "0.1 [H]"
    R' = "1.0 [ohm]"
    L = "0.1 [H]"
    C = "1.0 [uF]"
    L' = "0.0001 [H]"
    C' = "1.0 [uF]"
    IA = ""
    IB = ""
    IC = ""
    }
   0.const([468,324],0,0,60)
    {
    Name = ""
    Value = "7.967"
    }
   0.breaker3([540,396],3,0,80)
    {
    Ctrl = "0"
    OPCUR = "0"
    ENAB = "0"
    CLVL = "0.0 [kA]"
    View = "1"
    ViewB = "0"
    DisPQ = "0"
    NAME = "GRID"
    NAMEA = "BRKA"
    NAMEB = "BRKB"
    NAMEC = "BRKC"
    ROFF = "1.0E6 [ohm]"
    RON = "0.01 [ohm]"
    PRER = "0.5 [ohm]"
    TDA = "0.0 [s]"
    TDB = "0.0 [s]"
    TDC = "0.0 [s]"
    TDRA = "0.05 [s]"
    TDRB = "0.05 [s]"
    TDRC = "0.05 [s]"
    PostIns = "0"
    TDBOA = "0.005 [s]"
    IBRA = ""
    IBRB = ""
    IBRC = ""
    IBR0 = ""
    SBRA = ""
    SBRB = ""
    SBRC = ""
    BP = ""
    BQ = ""
    BOpen1 = "0"
    BOpen2 = "0"
    BOpen3 = "0"
    P = "0 [MW]"
    Q = "0 [MVAR]"
    }
   -Wire-([432,432],0,0,-1)
    {
    Vertex="0,0;108,0"
    }
   0.compare([468,90],0,0,20)
    {
    X = "3"
    OL = "1"
    OH = "0"
    INTR = "0"
    }
   0.datalabel([540,90],3,0,-1)
    {
    Name = "GRID"
    }
   0.time-sig([396,90],0,0,10)
    {
    }
   0.unity([540,90],0,0,30)
    {
    IType = "2"
    OType = "1"
    }
   0.nl_tfun([396,288],0,0,210)
    {
    N = "4"
    x1 = "0"
    y1 = "60"
    x2 = "10"
    y2 = "60"
    x3 = "15"
    y3 = "59.0"
    x4 = "20"
    y4 = "59.0"
    x5 = "0.0"
    y5 = "0.0"
    x6 = "1.0"
    y6 = "1.0"
    x7 = "2.0"
    y7 = "2.0"
    x8 = "3.0"
    y8 = "3.0"
    x9 = "4.0"
    y9 = "4.0"
    x10 = "5.0"
    y10 = "5.0"
    }
   -Wire-([432,288],0,0,-1)
    {
    Vertex="0,0;72,0"
    }
   0.time-sig([324,360],0,0,70)
    {
    }
   -Wire-([360,288],0,0,-1)
    {
    Vertex="0,0;0,72"
    }
   0.pss([72,324],0,0,230)
    {
    INIT_P = "InitEx"
    INIT_D = "InitEx"
    PSS = "1"
    DEC = "0"
    VS1_1 = "2"
    T6_1 = "0.0 [s]"
    KS_1 = "10.0 [pu]"
    T5_1 = "10.0 [s]"
    A1_1 = "0.0"
    A2_1 = "0.0"
    T1_1 = "1.5 [s]"
    T2_1 = "1.0 [s]"
    T3_1 = "1 [s]"
    T4_1 = "1 [s]"
    VSTX_1 = "0.1 [pu]"
    VSTN_1 = "-0.1 [pu]"
    VS1_2 = "0"
    VS1X_2 = "0.08 [pu]"
    VS1N_2 = "-0.08 [pu]"
    TW1_2 = "10.0 [s]"
    TW2_2 = "10.0 [s]"
    T6_2 = "0.0 [s]"
    VS2_2 = "0"
    VS2X_2 = "1.25 [pu]"
    VS2N_2 = "-1.25 [pu]"
    TW3_2 = "10.0 [s]"
    TW4_2 = "10.0 [s]"
    KS2_2 = "5.0 [pu]"
    T7_2 = "0.0 [s]"
    KS3_2 = "5.0 [pu]"
    T8_2 = "0.5 [s]"
    T9_2 = "0.1 [s]"
    N_2 = "1"
    M_2 = "1"
    KS1_2 = "5.0 [pu]"
    T1_2 = "0.0 [s]"
    T2_2 = "6.0 [s]"
    T3_2 = "0.08 [s]"
    T4_2 = "0.01 [s]"
    T10_2 = "0.08 [s]"
    T5_2 = "0.01 [s]"
    VSTX_2 = "0.1 [pu]"
    VSTN_2 = "-0.1 [pu]"
    VS1_3 = "0"
    T1_3 = "0.02 [s]"
    T2_3 = "1.5 [s]"
    KS1_3 = "1.0 [pu]"
    VS2_3 = "0"
    T3_3 = "0.02 [s]"
    T4_3 = "1.5 [s]"
    KS2_3 = "0.0 [pu]"
    VSTX_3 = "0.1 [pu]"
    VSTN_3 = "-0.1 [pu]"
    VS1_4 = "0"
    T1_4 = "0.02 [s]"
    T2_4 = "1.5 [s]"
    KS1_4 = "1.0 [pu]"
    VS2_4 = "0"
    T3_4 = "0.03 [s]"
    T4_4 = "1.5 [s]"
    KS2_4 = "0.0 [pu]"
    T0_4 = "0.2 [s]"
    M_4 = "1.5 [s]"
    VSTX_4 = "0.1 [pu]"
    VSTN_4 = "-0.1 [pu]"
    TW5_1 = "5.0 [s]"
    VTC_1 = "0.95 [pu]"
    VAL_1 = "5.5 [pu]"
    ESC_1 = "0.0015 [pu]"
    VTLMT = "1.1 [pu]"
    VOMX_1 = "0.3 [pu]"
    VOMN_1 = "0.1 [pu]"
    KETL_1 = "47.0 [pu]"
    TL1_1 = "0.025 [s]"
    TL2_1 = "1.25 [s]"
    VTM_1 = "1.13 [pu]"
    VTN_1 = "1.12 [pu]"
    TD_1 = "0.03 [s]"
    KAN_1 = "400.0 [pu]"
    TAN_1 = "0.08 [s]"
    VAMX_1 = "1.0 [pu]"
    VSMX_1 = "0.2 [pu]"
    VSMN_1 = "-0.066 [pu]"
    VK_2 = "0.08 [pu]"
    TD1_2 = "0.10 [s]"
    TD2_2 = "10.0 [s]"
    VTMX_2 = "1.1 [pu]"
    VDMX_2 = "1.1 [pu]"
    VDMN_2 = "1.1 [pu]"
    VTMN_3 = "0.95 [pu]"
    TDR_3 = "1.0 [s]"
    }
   0.datalabel([36,432],0,0,-1)
    {
    Name = "W"
    }
   0.pgb([126,288],3,106931016,400)
    {
    Name = "MECHANICAL TORQUE"
    Group = "Synchronous Machine"
    Display = "0"
    Scale = "1.0"
    Units = "pu"
    mrun = "0"
    Pol = "0"
    Min = "0"
    Max = "1"
    }
   -Wire-([108,288],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.gain([36,396],3,0,220)
    {
    G = "0.0026525"
    COM = "Gain"
    Dim = "1"
    }
   -Plot-([612,0],0)
    {
    Title = "$(GROUP) : Graphs"
    Draw = 1
    Area = [0,0,0,0]
    Posn = [612,0]
    Icon = [-1,-1]
    Extents = 0,0,576,288
    XLabel = " "
    AutoPan = "false,75"
    Graph([0,0],[0,0,576,225],"y")
     {
     Options = 128
     Units = ""
     Curve(106931016,"MECHANICAL TORQUE",0,,,)
     }
    }
   -Plot-([594,126],0)
    {
    Title = ""
    Draw = 1
    Area = [0,0,558,1116]
    Posn = [594,126]
    Icon = [-1,-1]
    Extents = 0,0,558,1116
    XLabel = ""
    AutoPan = "false,75"
    Graph([0,0],[0,0,558,151],"Power")
     {
     Options = 0
     Units = "MW"
     Curve(106882464,"REAL POWER",0,,,)
     }
    Graph([0,151],[0,0,558,151],"Reactive Power")
     {
     Options = 0
     Units = "MVAR"
     Curve(106890624,"REACTIVE POWER",0,,,)
     }
    Graph([0,302],[0,0,558,151],"Speed")
     {
     Options = 2
     Units = "rad/s"
     Curve(106880832,"OMEGA",0,,,)
     }
    Graph([0,453],[0,0,558,150],"Torque")
     {
     Options = 0
     Units = "pu"
     Curve(106883280,"MECHANICAL TORQUE",0,,,)
     }
    Graph([0,603],[0,0,558,150],"Current")
     {
     Options = 0
     Units = "pu"
     Curve(106882872,"FIELD CURRENT",0,,,)
     }
    Graph([0,753],[0,0,558,150],"Voltage")
     {
     Options = 0
     Units = "pu"
     Curve(106886544,"FIELD VOLTAGE",0,,,)
     }
    Graph([0,903],[0,0,558,150],"Voltage")
     {
     Options = 0
     Units = "kV"
     Curve(106873488,"TERMINAL VOLTAGE",0,,,)
     }
    }
   }
  }
 }

