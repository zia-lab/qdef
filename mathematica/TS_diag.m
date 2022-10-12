(* -----------------------------------------------------------------
This  is  a quick and dirty script to produce Tanabe-Sugano diagrams
don't expect the code to make much sense.
For complaints reach out to David.
----------------------------------------------------------------- *)

BeginPackage["tsDIAG`"]

SetDirectory[NotebookDirectory[]];
cowandata = Import["brik_and_ma_from_cowan.xlsx"][[1]];
h5fname = "tsk_hypercube_int16.h5.gz";
normalizer = 1;
If[h5fname == "tsk_hypercube_int16.h5.gz",
  normalizer = 100.];
headers = cowandata[[1]];
cowandata = cowandata[[2 ;;]];
evtocm = 8065.544290;
cowanA = Association[
   {#[[1]], Round[#[[2]]]} -> Association[{
        "d^n" -> Round[#[[4]]],
        "B" -> Round[#[[6]]],
        "C" -> #[[7]],
        "\[Zeta]" -> #[[11]],
        "C/B" -> Round[#[[7]]/#[[6]], 0.1]
        }] & /@ cowandata];
tskkeys = Import[h5fname];
\[Gamma]soB = Import[h5fname, "/params/gammas_B"]/normalizer;
DqsoB = Import[h5fname, "/params/Dqs_B"]/normalizer;
\[Zeta]soB = Import[h5fname, "/params/zetas_B"]/normalizer;
keyTemplate = 
  StringTemplate["/`num_electrons`/`gamma_idx`/`zeta_idx`"];
Off[Nearest::neard];
Off[Part::partw];
shellToatoms = <|
   "3d" -> StringSplit["Sc Ti V Cr Mn Fe Co Ni Cu Zn", " "],
   			"4d" -> StringSplit["Y Zr Nb Mo Tc Ru Rh Pd Ag Cd", " "],
   			"5d" -> StringSplit["Lu Hf Ta W Re Os Ir Pt Au Hg", " "]|>;
evtocm = 8065.54429;

allMinima = Association[];
TSKDiagram[nephRatio0_, Erange0_, DqrangeR0_, shell0_, atom0_, charge0_, 
  vLine0_, DqLine0_, positiveOnly0_, p0_ : False] := (Module[
    {nephRatio=nephRatio0,
    Erange=Erange0,
    DqrangeR=DqrangeR0,
    shell=shell0,
    atom=atom0,
    charge=charge0, 
    vLine=vLine0,
    DqLine=DqLine0,
    positiveOnly=positiveOnly0,
    p=p0},
    (
   key = {atom, charge};
   If[
    MemberQ[Keys[cowanA], key],
    (
     \[Gamma] = cowanA[key]["C/B"];
     B = Round[nephRatio*cowanA[key]["B"], 1];
     \[Zeta] = cowanA[key]["\[Zeta]"];
     \[Zeta] = \[Zeta]/B;
     Dqrange = Min[6*B/evtocm, DqrangeR];
     \[Gamma]oB = Nearest[\[Gamma]soB, \[Gamma]][[1]];
     \[Zeta]oB = Nearest[\[Zeta]soB, \[Zeta]][[1]];
     \[Gamma]idx = Position[\[Gamma]soB, \[Gamma]oB][[1, 1]] - 1;
     \[Zeta]idx = Position[\[Zeta]soB, \[Zeta]oB][[1, 1]] - 1;
     numElectrons = cowanA[key]["d^n"];
     tskkey = 
      keyTemplate[<|"num_electrons" -> numElectrons, 
        "gamma_idx" -> \[Gamma]idx, "zeta_idx" -> \[Zeta]idx|>];
     If[MemberQ[tskkeys, tskkey], (
       data = Import[h5fname, tskkey]/normalizer;
       If[KeyExistsQ[allMinima, tskkey],
        minima = allMinima[tskkey];
        ,
        (minima = Min /@ data;
         allMinima[tskkey] = minima;)
        ];
       data = data - minima;
       data = Transpose[data];
       data = Transpose[{DqsoB*B/evtocm, #*B/evtocm}] & /@ data;
       rdata = data;
       
       lrange = If[positiveOnly, 0, -Dqrange];
       
       
       yTicksLeft = Table[{ee, ee}, {ee, 0, Erange, 0.2}];
       
       eVstep = N[10^(Ceiling[Log10[Dqrange]] - 1)];
       eVstep = Max[0.01, eVstep];
       xTicksBottom1 = Table[{ee, ee}, {ee, 0, Dqrange, eVstep}];
       xTicksBottom2 = Table[{-ee, -ee}, {ee, 0, Dqrange, eVstep}];
       xTicksBottom = Join[xTicksBottom1[[2 ;;]], xTicksBottom2];
       
       step = 10^(Floor[Log10[Dqrange*evtocm]] - 1);
       If[positiveOnly,
        dTick = Round[Floor[Dqrange*evtocm/10], step],
        dTick = Round[Floor[Dqrange*evtocm/5], step]
        ];
       yTicksRight = 
        Table[{ee/evtocm, ee}, {ee, 0, Round[Erange*evtocm], 2000}];
       xTicksTop1 = 
        Table[{ee/evtocm, ee}, {ee, 0, Round[Dqrange*evtocm, 100], 
          dTick}];
       xTicksTop2 = 
        Table[{-ee/evtocm, -ee}, {ee, 0, Round[Dqrange*evtocm, 100], 
          dTick}];
       xTicksTop = Join[xTicksTop1[[2 ;;]], xTicksTop2];
       
       tdata = Transpose[data];
       multidata = Association[];
       multidata = Table[
         (Dq = datum[[1, 1]];
          eivals = Chop[Last /@ datum];
          eivals = Round[eivals, 0.001];
          uniqueEigenvals = DeleteDuplicates[eivals];
          
          Table[{Dq, eival, Count[eivals, eival]}, {eival, 
            uniqueEigenvals}]
          )
         , {datum, tdata}];
       multidata = Flatten[multidata, 1];
       multis = Sort[DeleteDuplicates[Last /@ multidata]];
       data = 
        Table[{#[[1]], #[[2]]} & /@ 
          Select[multidata, #[[3]] == count &], {count, multis}];
       {\[Gamma]oB, \[Zeta]oB};
       plotLabel = 
        Row[{Superscript[atom, ToString[charge] <> "+"], 
          " | \[Beta]=", ToString[nephRatio] , " | ", 
          Superscript[shell, numElectrons], 
          " | \!\(\*SubscriptBox[\(B\), \(HF\)]\)*\[Beta]=" <> 
           ToString[B] <> " \!\(\*SuperscriptBox[\(cm\), \(-1\)]\)", 
          " | \[Zeta]=" <> ToString[\[Zeta]*B] <> 
           " \!\(\*SuperscriptBox[\(cm\), \(-1\)]\)", 
          " | C/B=" <> ToString[\[Gamma]oB]}, BaseStyle -> "Title"];
       epiGraphs = {
         {Blue, Dashed, InfiniteLine[{DqLine, 0}, {0, 1}]},
         {Blue, Dashed, InfiniteLine[{0, vLine}, {1, 0}]},
         {Gray, Dashed, Line[{{-6*B/evtocm, 0}, {6*B/evtocm, 0}}]}
         };
       If[Not[p === False],
        extraEpi = {Tooltip[Point[p], Column[{p, evtocm*p}]],
          {Red, Dashed, 
           Line[{{-6*B/evtocm, p[[2]]}, {6*B/evtocm, p[[2]]}}]},
          {Red, Dashed, Line[{{p[[1]], 0}, {p[[1]], Erange}}]}
          },
        extraEpi = {}
        ];
       epiGraphs = Join[epiGraphs, extraEpi];
       p1 = ListPlot[data,
         Frame -> {{True, True}, {True, True}},
         
         FrameLabel -> {{"E/eV", 
            "E/\!\(\*SuperscriptBox[\(cm\), \(-1\)]\)"}, {"Dq/eV", 
            "Dq/\!\(\*SuperscriptBox[\(cm\), \(-1\)]\)"}},
         FrameStyle -> Directive[16],
         
         FrameTicks -> {{yTicksLeft, yTicksRight}, {xTicksBottom, 
            xTicksTop}},
         ImageSize -> 1200,
         Joined -> False,
         PlotStyle -> Automatic,
         PlotRange -> {
           {If[positiveOnly, 0, -Dqrange], Dqrange},
           {-0.1, Erange*evtocm/B*B/evtocm}
           },
         PlotLegends -> multis,
         PlotLabel -> plotLabel,
         Epilog -> epiGraphs
         ];
       p2 = ListPlot[rdata,
         PlotStyle -> Directive[Thickness[0.001],
           Opacity[0.03],
           Black,
           Dashed],
         Joined -> True, PlotLegends -> None];
       Return[Show[p1, p2]]
       ),
      Return["MissingB"]
      ]
     ),
    Return["MissingA"]
    ]
   )
  ]
  )

EndPackage[]
