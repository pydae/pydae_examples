PSCAD 4.2.0

Settings
 {
 Id = "1668301701.1668442250"
 Author = "jmmau.jmmau"
 Desc = ""
 Arch = "windows"
 Options = 32
 Build = 18
 Warn = 1
 Check = 15
 Libs = ""
 Source = ""
 RunInfo = 
  {
  Fin = 0.5
  Step = 5e-005
  Plot = 0.00025
  Chat = 0.001
  Brch = 0.0005
  Lat = 100
  Options = 0
  Advanced = 4607
  Debug = 0
  StartFile = ""
  OFile = "noname.out"
  SFile = "noname.snp"
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
  Desc = ""
  FileDate = 1668302189
  Nodes = 
   {
   }

  Graphics = 
   {
   Rectangle(-18,-18,18,18)
   }


  Page(A/A4,Landscape,16,[640,323],5)
   {
   -Wire-([288,162],0,0,-1)
    {
    Vertex="0,0;126,0"
    }
   -Wire-([288,198],0,0,-1)
    {
    Vertex="0,0;126,0"
    }
   -Wire-([288,234],0,0,-1)
    {
    Vertex="0,0;126,0"
    }
   0.excac([252,90],0,0,140)
    {
    TYPE = "1"
    INIT = "InitEx"
    OVR = "0"
    RC = "0.0 [pu]"
    XC = "0.0 [pu]"
    TR = "0.0 [s]"
    STAB = "0"
    TC_1 = "0.0 [s]"
    TB_1 = "0.0 [s]"
    KA_1 = "400.0 [pu]"
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
   -Wire-([180,90],0,0,-1)
    {
    Vertex="0,0;0,36"
    }
   -Wire-([288,90],0,0,-1)
    {
    Vertex="0,0;0,36"
    }
   0.sync_machine([252,198],0,0,130)
    {
    Name = "Sync1"
    Nqaxw = "1"
    Cnfg = "0"
    MM = "1"
    CfRa = "1"
    MSat = "0"
    icTyp = "1"
    Iscl = "1"
    View = "0"
    itrfa = "0"
    izro = "0"
    Icsp = "0"
    Icmp = "0"
    Immd = "0"
    Ifmlt = "0"
    Term = "3"
    Ts = "0.02 [s]"
    Iexc = "1"
    Igov = "0"
    Ospd = "1"
    machsw = "0"
    Enab = "1"
    npadjs = "0"
    pset = "0.0 [MW]"
    nftrsw = "0"
    hmult = "1.0"
    sdmftr = "1.0"
    sdmspd = "1.0"
    npadjm = "0"
    fldmlt = "1.0"
    nfldsw = "0"
    Vbase = "0.4 [kV]"
    Ibase = "0.0288675 [kA]"
    OMO = "314.15926 [rad/s]"
    H = "5.0 [s]"
    D = "0.0 [pu]"
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
    Ra = "0.002 [pu]"
    Ta = "0.332 [s]"
    Xp = "0.130 [pu]"
    Xd = "0.920 [pu]"
    Xd' = "0.300 [pu]"
    Tdo' = "5.2 [s]"
    Xd'' = "0.220 [pu]"
    Tdo'' = "0.029 [s]"
    Gfld = "1.0E+2 [pu]"
    Xkf = "0.0 [pu]"
    Xq = "0.510 [pu]"
    Xq' = "0.228 [pu]"
    Tqo' = "0.85 [s]"
    Xq'' = "0.290 [pu]"
    Tqo'' = "0.034 [s]"
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
    VT = "1.0 [pu]"
    Pheta = "-0.095993 [rad]"
    Trmpv = "0.1 [s]"
    Sysfl = "100.0 [pu]"
    Ptcon = "0.2 [s]"
    P0 = "-25.0 [MW]"
    Q0 = "-270.0 [MVAR]"
    Theti = "3.141592 [rad]"
    Idi = "0.0 [pu]"
    Iqi = "0.0 [pu]"
    Ifi = "0.0 [pu]"
    Spdi = "1.0 [pu]"
    POut = ""
    QOut = ""
    Vneut = ""
    Cneut = ""
    Lang = ""
    Theta = ""
    Wang = ""
    Tesmt = "Testdy"
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
   0.nodeloop([486,198],0,0,50)
    {
    View = "0"
    }
   0.nodeloop([378,198],0,0,40)
    {
    View = "0"
    }
   -Wire-([378,270],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.nodeloop([576,198],0,0,60)
    {
    View = "0"
    }
   0.annotation([594,396],1,0,-1)
    {
    AL1 = "Three Phase"
    AL2 = "RMS Voltage Meter"
    }
   -Wire-([450,270],0,0,-1)
    {
    Vertex="0,0;36,0"
    }
   0.annotation([432,450],2,0,-1)
    {
    AL1 = "Instantaneous Real and Reactive"
    AL2 = "Power Meter"
    }
   0.rms3ph([576,306],1,0,100)
    {
    Type = "0"
    View = "0"
    Scale = "230.0 [kV]"
    Ts = "0.02 [s]"
    freq = "60.0 [Hz]"
    NSAM = "64"
    Vinit = "0.0 [kV]"
    }
   0.power([450,306],1,0,90)
    {
    DIR = "1"
    P = "1"
    Q = "1"
    TS = "0.02 [s]"
    View = "0"
    }
   0.resistor([450,162],2,0,-1)
    {
    R = "0.01 [ohm]"
    }
   0.resistor([450,198],2,0,-1)
    {
    R = "0.01 [ohm]"
    }
   0.resistor([450,234],2,0,-1)
    {
    R = "0.01 [ohm]"
    }
   -Wire-([450,162],0,0,-1)
    {
    Vertex="0,0;252,0"
    }
   -Wire-([450,198],0,0,-1)
    {
    Vertex="0,0;252,0"
    }
   -Wire-([450,234],0,0,-1)
    {
    Vertex="0,0;252,0"
    }
   0.source_3([738,198],4,0,70)
    {
    Name = "Source 1"
    Type = "4"
    Grnd = "1"
    View = "0"
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
    R = "1.0 [ohm]"
    Rs = "1.0 [ohm]"
    Rp = "1.0 [ohm]"
    Lp = "0.1 [H]"
    R' = "1.0 [ohm]"
    L = "0.1 [H]"
    C = "1.0 [uF]"
    L' = "0.1 [H]"
    C' = "1.0 [uF]"
    IA = ""
    IB = ""
    IC = ""
    }
   0.const([216,18],0,0,10)
    {
    Name = ""
    Value = "1"
    }
   0.const([702,126],0,0,20)
    {
    Name = ""
    Value = "231"
    }
   0.const([810,126],2,0,30)
    {
    Name = ""
    Value = "50"
    }
   0.const([288,270],2,0,80)
    {
    Name = ""
    Value = "0"
    }
   0.pgb([450,342],1,106877976,120)
    {
    Name = "<Untitled>"
    Group = ""
    Display = "0"
    Scale = "1.0"
    Units = ""
    mrun = "0"
    Pol = "0"
    Min = "-2.0"
    Max = "2.0"
    }
   0.pgb([414,342],1,106874304,110)
    {
    Name = "<Untitled>"
    Group = ""
    Display = "0"
    Scale = "1.0"
    Units = ""
    mrun = "0"
    Pol = "0"
    Min = "-2.0"
    Max = "2.0"
    }
   -Plot-([180,486],0)
    {
    Title = "$(GROUP) : Graphs"
    Draw = 1
    Area = [0,0,0,0]
    Posn = [180,486]
    Icon = [-1,-1]
    Extents = 0,0,576,288
    XLabel = " "
    AutoPan = "false,75"
    Graph([0,0],[0,0,576,225],"y")
     {
     Options = 128
     Units = ""
     Curve(106877976,"<Untitled>",0,,,)
     }
    }
   }
  }
 }

