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

LeftSuperscriptRightSubscript[a_, x_, b_] := 
  Row[{Superscript["\[InvisibleSpace]", a], Subscript[x, b]}];
LeftSuperscript[a_, x_] := 
  Row[{Superscript["\[InvisibleSpace]", a], x}];
ParseTermSymbol[termString_] := (
  bits = StringSplit[termString, ","];
  If[StringContainsQ[bits[[2]], "E" ],
   Return[LeftSuperscript[bits[[1]], bits[[2]]]],
   (chunks = StringSplit[bits[[2]], "_"];
    Return[
     LeftSuperscriptRightSubscript[bits[[1]], chunks[[1]], 
      chunks[[2]]]])
   ]
  )
fname = "./tsk_diag.h5";
dset = Table[
   nume -> (<|"tskData" -> Import[fname, ToString[nume]],
      "terms" -> (StringSplit[#, ","] & /@ 
         Import[fname, ToString[nume] <> "-labels"]),
      "tskLabels" -> 
       ParseTermSymbol /@ Import[fname, ToString[nume] <> "-labels"],
      "Dqs" -> Import[fname, "Dqs"],
      "\[Gamma]s" ->  Import[fname, "gammas"]|>), {nume, 1, 5}];
dset = Association[dset];
DqsCub = dset[[2]]["Dqs"];
gammas = dset[[2]]["\[Gamma]s"];
cuts = Table[numElectrons -> Table[(
      pdqsp = 
       DqsCub[[Flatten[
          1 + Position[
            Differences[
             Ordering[#][[1]] & /@ 
              Transpose[
               dset[[numElectrons]]["tskData"][[idx]]]], _?(# != 
                0 &)]]]];
      pdqsm = 
       DqsCub[[Flatten[
          0 + Position[
            Differences[
             Ordering[#][[1]] & /@ 
              Transpose[
               dset[[numElectrons]]["tskData"][[idx]]]], _?(# != 
                0 &)]]]];
      pdqs = (pdqsp + pdqsm)/2;
      pdqs = Select[pdqs, Abs[#] > 0.2 &]), {idx, 1, Length[gammas], 
      1}], {numElectrons, 1, 5}];
cuts = Association[cuts];
gammaFun[gamma_] := Ordering[Abs[gammas - gamma]][[1]];

energyStart = 1000;
energyDiv = 100;
energyUnit = "\!\(\*SuperscriptBox[\(cm\), \(-1\)]\)";

multiPs = <|
   1 -> {"2"}, 
   2 -> {"1", "3"}, 
   3 -> {"2", "4"}, 
   4 -> {"1", "3", "5"},
   5 -> {"2", "4", "6"},
   6 -> {"1", "3", "5"},
   7 -> {"2", "4"},
   8 -> {"1", "3"},
   9 -> {"2"}|>;

allMinima = Association[];
TSKDiagram[nephRatio0_, Erange0_, DqrangeR0_, shell0_, atom0_, charge0_, 
  vLine0_, DqLine0_, positiveOnly0_, \[Chi]0_, p0_ : False] := (Module[
    {nephRatio=nephRatio0,
    Erange=Erange0,
    DqrangeR=DqrangeR0,
    shell=shell0,
    atom=atom0,
    charge=charge0, 
    vLine=vLine0,
    DqLine=DqLine0,
    positiveOnly=positiveOnly0,
    \[Chi] = \[Chi]0,
    p=p0},
    (
   key = {atom, charge};
   If[
    MemberQ[Keys[cowanA], key],
    (
     \[Gamma] = cowanA[key]["C/B"];
     B = Round[nephRatio*cowanA[key]["B"], 1];
     \[Zeta] = cowanA[key]["\[Zeta]"];
     \[Zeta] = \[Chi]*\[Zeta]/B;
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

TSKDiagramNoSO[shellR_,
  atomR_,
  chargeR_,
  spinR_,
  DqrangeR_,
  ErangeR_,
  showLabelsR_,
  vLineR_,
  extraUnitR_] :=
 Module[{shell = shellR,
   atom = atomR,
   charge = chargeR,
   spin = spinR,
   Dqrange = DqrangeR,
   Erange = ErangeR,
   showLabels = showLabelsR,
   vLine = vLineR,
   extraUnit = extraUnitR},
  (
   key = {atom, charge};
   If[(MemberQ[Keys[cowanA], key]), (
     numElectrons = cowanA[{atom, charge}]["d^n"];
     fieldTypemultiplier = 1;
     Dqs = DqsCub;
     gamma = cowanA[{atom, charge}]["C/B"];
     B = Round[cowanA[key]["B"], 1];
     idx = gammaFun[gamma];
     \[CapitalNu] = 
      If[numElectrons > 5, 10 - numElectrons, numElectrons];
     (*Sign inversion for hole-electron equivalency*)
     sign = If[numElectrons > 5, -1, 1];
     If[(MemberQ[Keys[dset], \[CapitalNu]] && NumericQ[B]), (
       multiplicities = 
        StringJoin[
         Riffle[DeleteDuplicates[
           First /@ dset[\[CapitalNu]]["terms"]], ","]];
       plotTitle = Row[{"| ",
          multiplicities,
          " | ",
          Superscript["d", numElectrons],
          "| C/B \[TildeTilde] " <> ToString[gammas[[idx]]],
          " | B \[TildeTilde] " <> ToString[B] <> " " <> energyUnit,
          " | ",
          Superscript[atom, ToString[charge] <> "+"]
          }, BaseStyle -> "Title"];
       plotTitle = Framed[plotTitle];
       legends = dset[\[CapitalNu]]["tskLabels"];
       mask = (MemberQ[spin, #[[1]]]) & /@ dset[\[CapitalNu]]["terms"];
       legends = Pick[legends, mask];
       maskedData = Pick[dset[\[CapitalNu]]["tskData"][[idx]], mask];
       Dqindices = 
        First /@ 
         Select[Transpose[{Range[Length[Dqs]], Dqs}], 
          Abs[#[[2]]] <= Dqrange &];
       rightMarginValues = #[[Dqindices[[-1]]]] & /@ maskedData;
       leftMarginValues = #[[Dqindices[[1]]]] & /@ maskedData;
       
       maskedData = (#[[Dqindices]] & /@ maskedData);
       signedDqs = sign*Dqs;
       signedDqs = signedDqs[[Dqindices]];
       plotData = 
        Tooltip[#[[1]], #[[2]]] & /@ 
         Transpose[{(Transpose[{signedDqs, #}] & /@ maskedData), 
           legends}];
       If[extraUnit == "\!\(\*SuperscriptBox[\(cm\), \(-1\)]\)", (
         (*right frame labels*)
         
         extraTicks = 
          Table[{ee/B, ee}, {ee, 0, B*Erange, 
            Round[B*Erange/10, 10^(Floor[Log10[B*Erange]] - 2)]}];
         (*top frame labels*)
         
         moTicks = 
          Table[{ee/B, ee}, {ee, 0, B*Dqrange, 
            Round[B*Dqrange/10, 10^(
             Floor[Log10[B*Dqrange]] - 2)]}];),
        (
         (*right frame labels*)
         
         extraTicks = 
          Table[{ee/B*evtocm, ee}, {ee, 0, B*Erange/evtocm, 0.5}];
         (*top frame labels*)
         
         moTicks = 
          Table[{ee/B*evtocm, ee}, {ee, 0, B*Dqrange/evtocm, 0.1}];
         )
        ];
       moTicks = Join[{-#[[1]], -#[[2]]} & /@ moTicks, moTicks];
       If[showLabels, (
         
         labelonRightMargin = 
          Flatten[Position[rightMarginValues, x_ /; Abs[x] <= Erange]];
         
         labelonLeftMargin = 
          Flatten[Position[leftMarginValues, 
            x_ /; Abs[x] <= Erange]];
         rightMarginLabels = legends[[labelonRightMargin]];
         leftMarginLabels = legends[[labelonLeftMargin]];
         
         rightCallouts = 
          Callout[{sign*#[[1]][[1]], #[[1]][[2]]}, #[[2]], Right, 
             Appearance -> "Frame", LabelVisibility -> All] & /@ 
           Transpose[{{Dqrange, #} & /@ 
              rightMarginValues[[labelonRightMargin]], 
             rightMarginLabels}];
         
         leftCallouts = 
          Callout[{sign*#[[1]][[1]], #[[1]][[2]]}, #[[2]], Left, 
             Appearance -> "Frame", LabelVisibility -> All] & /@ 
           Transpose[{{-Dqrange, #} & /@ 
              leftMarginValues[[labelonLeftMargin]], 
             leftMarginLabels}];
         callouts = Join[rightCallouts, leftCallouts];
         
         calloutPlot = 
          ListPlot[callouts, 
           PlotRange -> {{-Dqrange, Dqrange}, {-1, Erange}}, 
           PlotStyle -> PointSize[Small]];
         )];
       
       tskfig = ListPlot[plotData,
         ImageSize -> 1000,
         Joined -> True,
         PlotLabel -> plotTitle,
         Frame -> True,
         FrameLabel -> {{"E/B", "E/" <> extraUnit},
           {"Dq/B",
            "Dq/" <> extraUnit}},
         FrameStyle -> Directive[15],
         FrameTicks -> {{Automatic, extraTicks}, {Automatic, moTicks}},
         PlotRange -> {{-Dqrange*1.1, Dqrange*1.1}, {-1, Erange}},
         PlotLegends -> None,
         Epilog -> Flatten[{
            ({Purple, Dashed, 
                Line[{{sign*#, 0}, {sign*#, Erange}}]}) & /@ 
             cuts[[\[CapitalNu]]][[idx]],
            Line[{{Dqrange, 0}, {Dqrange, Erange}}],
            Line[{{-Dqrange, 0}, {-Dqrange, Erange}}],
            {Thick, Orange, Dashed, 
             Line[{{-Dqrange, vLine*8065.54429/B}, {Dqrange, 
                vLine*8065.54429/B}}]}
            }],
         InterpolationOrder -> 1];
       If[showLabels,
        Return[Show[tskfig, calloutPlot]],
        Return[tskfig]
        ]),
      Return["Not a valid ion, try a different charge."]
      ]
     ),
    Return["Missing"]]
   )
  ]

EndPackage[]
