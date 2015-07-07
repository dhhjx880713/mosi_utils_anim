

//simplified test application using parts of MPlan3D
(function() {
	
	 window.addEventListener("load", onLoad, false);

    var quat = XML3D.math.quat;
    var vec3 = XML3D.math.vec3;
    var vec4 = XML3D.math.vec4;
    var mat4 = XML3D.math.mat4;
	function onLoad() {
		 //note if the application is initialized using 	a connect with the  DOM ready event	"  $( document ).ready( animationPlayer.onReady );  "
		//DOM element attributes cannot be access as properties of the javascript object, e.g. document.getElementById("defaultViewNode").transform is undefined
		
			  var rootNode = $("#root")[0]
			  var group = document.createElement("group");
			  group.setAttribute("id", "click_"+this.click_counter);
			  group.setAttribute("shader", "#Material_blue");
			  group.setAttribute("transform","#click_transform");
			  rootNode.appendChild(group);
			  skeletonPose = new SkeletonPose();
			  description = "		{       \"skeleton\":     {         \"joints\": [                    {                        \"name\": \"RightShoulder\",                        \"opposite\": \"LeftShoulder\",                        \"parent\": \"Neck\",                        \"zero\": [                            0.12506141884222691,                            -0.61352943311603014,                            0.11395940754115953,                            0.77133295641399879,                            0.86415573612225827,                            0.30022353452578976,                            4.6718976077279013,                            -0.59155160840700627                        ]                    },                    {                        \"name\": \"LeftLeg\",                        \"opposite\": \"RightLeg\",                        \"parent\": \"LeftUpLeg\",                        \"zero\": [                            0.99809603520891066,                            0.0,                            0.0,                            -0.06167904425534694,                            -7.610072692584324e-07,                            19.913786070358935,                            1.2306072415283222,                            -1.2314690465492345e-05                        ]                    },                    {                        \"name\": \"LeftHand\",                        \"opposite\": \"RightHand\",                        \"parent\": \"LeftForeArm\",                        \"zero\": [                            0.69832815195686082,                            -0.71476224963173718,                            0.012448761055246041,                            0.036024256190560898,                            9.6821530625171093,                            9.4595371173557528,                            -0.48798424828830811,                            0.16863012073347902                        ]                    },                    {                        \"name\": \"Spine\",                        \"opposite\": null,                        \"parent\": \"Hips\",                        \"zero\": [                            0.99975230388057579,                            -2.0642524305366003e-06,                            -6.7779596672391996e-07,                            -0.022256030211162948,                            1.2609240976387584e-05,                            6.013299514395845,                            0.052417665515597373,                            7.0821747825474009e-06                        ]                    },                    {                        \"name\": \"RightLeg\",                        \"opposite\": \"LeftLeg\",                        \"parent\": \"RightUpLeg\",                        \"zero\": [                            0.99809603498428967,                            0.0,                            0.0,                            -0.061679047890183303,                            -7.4963995031804389e-07,                            19.913782523263105,                            1.230600028086122,                            -1.2130742734719538e-05                        ]                    },                    {                        \"name\": \"LeftShoulder\",                        \"opposite\": \"RightShoulder\",                        \"parent\": \"Neck\",                        \"zero\": [                            0.12505971984034614,                            0.61352977462336244,                            -0.11396155019684859,                            0.77133264367637866,                            -0.86415826197872847,                            0.30022002821033433,                            4.6718951270622977,                            0.5915656217501406                        ]                    },                    {                        \"name\": \"REyeBlinkTop\",                        \"opposite\": \"LEyeBlinkTop\",                        \"parent\": \"Head\",                        \"zero\": [                            0.81371577820447372,                            -0.10190217247938103,                            0.10861626356906059,                            0.56185860039102953,                            1.0834636377736651,                            7.3806141932096319,                            0.26968086251784584,                            -0.2826735484324141                        ]                    },                    {                        \"name\": \"Spine2\",                        \"opposite\": null,                        \"parent\": \"Spine1\",                        \"zero\": [                            0.99987888340530762,                            6.7173601746585677e-10,                            -4.3156116663510564e-08,                            0.015563371105046339,                            -8.6907334506076807e-08,                            7.6577700413949223,                            -0.12529076485752971,                            4.9054761500601198e-06                        ]                    },                    {                        \"name\": \"RightHandMiddle2\",                        \"opposite\": \"LeftHandMiddle2\",                        \"parent\": \"RightHandMiddle1\",                        \"zero\": [                            0.99778029687837244,                            -9.9229632878896184e-10,                            1.4868083761684785e-08,                            0.06659188510101427,                            -8.7692672773952497e-08,                            1.8964851023915861,                            -0.12657213011707993,                            1.370464001329856e-06                        ]                    },                    {                        \"name\": \"RightHand\",                        \"opposite\": \"LeftHand\",                        \"parent\": \"RightForeArm\",                        \"zero\": [                            0.69832813554893403,                            0.71476226413445976,                            -0.012448768913312253,                            0.036024283791263226,                            -9.6821515645038865,                            9.4595352912864268,                            -0.48798378821553112,                            -0.16863096475856754                        ]                    },                    {                        \"name\": \"LeftHandIndex2\",                        \"opposite\": \"RightHandIndex2\",                        \"parent\": \"LeftHandIndex1\",                        \"zero\": [                            0.99562210717940935,                            -1.1126943206823083e-08,                            -1.0446072982171097e-09,                            0.093469886571197341,                            2.4263570508675945e-08,                            1.5576328684491394,                            -0.14622795672396399,                            -7.4659398657890644e-08                        ]                    },                    {                        \"name\": \"Hips\",                        \"opposite\": null,                        \"parent\": \"Root\",                        \"zero\": [                            0.50000000000000022,                            0.5,                            0.49999999999999978,                            0.5,                            -22.379499435424801,                            22.379499435424805,                            22.379499435424815,                            -22.379499435424798                        ]                    },                    {                        \"name\": \"ROuterEyebrow\",                        \"opposite\": \"LOuterEyebrow\",                        \"parent\": \"Head\",                        \"zero\": [                            0.70899847628241486,                            -0.18039093173428608,                            0.18558627043968004,                            0.65600153094450953,                            1.9776120307350906,                            7.6232345450470431,                            -0.72942682525357916,                            0.1652588673725508                        ]                    },                    {                        \"name\": \"RightHandThumb2\",                        \"opposite\": \"LeftHandThumb2\",                        \"parent\": \"RightHandThumb1\",                        \"zero\": [                            0.99363449345380406,                            0.0,                            0.0,                            0.11265209016614888,                            4.4813985956593877e-07,                            1.4049082296341495,                            -0.15927835415719879,                            -3.9527648506078586e-06                        ]                    },                    {                        \"name\": \"LeftHandMiddle1\",                        \"opposite\": \"RightHandMiddle1\",                        \"parent\": \"LeftHand\",                        \"zero\": [                            0.97923375970029547,                            -0.13920087497550512,                            -0.028465963835955347,                            0.14461690486304807,                            0.65216766382430347,                            4.6689811679674689,                            -0.79261250127106231,                            -0.077866473384806956                        ]                    },                    {                        \"name\": \"RightUpLeg\",                        \"opposite\": \"LeftUpLeg\",                        \"parent\": \"Spine\",                        \"zero\": [                            0.02645970275836412,                            0.0018708038873279089,                            0.99007531332127774,                            -0.13801180447529807,                            -0.41417600902436347,                            4.2412349941744267,                            -0.84273221770584028,                            -6.0675448947932766                        ]                    },                    {                        \"name\": \"RightHandMiddle1\",                        \"opposite\": \"LeftHandMiddle1\",                        \"parent\": \"RightHand\",                        \"zero\": [                            0.97923376656107552,                            0.13920085844323959,                            0.02846596972570907,                            0.1446168731609464,                            -0.65216641033019596,                            4.668974558551187,                            -0.79260391252170437,                            0.077863271031121881                        ]                    },                    {                        \"name\": \"LeftHandThumb2\",                        \"opposite\": \"RightHandThumb2\",                        \"parent\": \"LeftHandThumb1\",                        \"zero\": [                            0.9936344869096374,                            -2.591103659480359e-08,                            -2.9376334707265155e-09,                            0.11265214788818233,                            2.0159986875309361e-07,                            1.4049145141212764,                            -0.15927854796698954,                            -1.4591971465861785e-06                        ]                    },                    {                        \"name\": \"RMouthCorner\",                        \"opposite\": \"LMouthCorner\",                        \"parent\": \"Head\",                        \"zero\": [                            0.65090786009372759,                            -0.15812989633195329,                            0.20154573514098909,                            0.71462802226104216,                            0.23287927877434367,                            5.4361881625107999,                            2.2472349894307522,                            0.35699732118688243                        ]                    },                    {                        \"name\": \"RightHandThumb1\",                        \"opposite\": \"LeftHandThumb1\",                        \"parent\": \"RightHand\",                        \"zero\": [                            0.66086999546564162,                            -0.65393942954482009,                            -0.36675115994257024,                            -0.033281500275939985,                            1.0434149981840752,                            1.4602976753036423,                            -0.79361975359847525,                            0.77149170973945336                        ]                    },                    {                        \"name\": \"LMouthCorner\",                        \"opposite\": \"RMouthCorner\",                        \"parent\": \"Head\",                        \"zero\": [                            0.65090712485313884,                            0.15813295223489637,                            -0.20155298191915422,                            0.71462597189886223,                            -0.23286694594040869,                            5.4361863414276606,                            2.247239271880991,                            -0.35700866538928133                        ]                    },                    {                        \"name\": \"Neck\",                        \"opposite\": null,                        \"parent\": \"Spine2\",                        \"zero\": [                            0.98570937631678046,                            4.7474474331514631e-14,                            -4.6722630545441012e-07,                            0.16845481720914188,                            -9.0038947977670957e-07,                            9.9464770498855604,                            -0.90680195613947545,                            2.7534988923906539e-06                        ]                    },                    {                        \"name\": \"RightFoot\",                        \"opposite\": \"LeftFoot\",                        \"parent\": \"RightLeg\",                        \"zero\": [                            0.9987290879261993,                            -0.014936452969592489,                            0.019717985522220493,                            0.043912553441295098,                            0.29625710859441817,                            19.809244586765715,                            -0.87099161388747603,                            0.39108311774198362                        ]                    },                    {                        \"name\": \"RightHandIndex1\",                        \"opposite\": \"LeftHandIndex1\",                        \"parent\": \"RightHand\",                        \"zero\": [                            0.9942351788501218,                            0.00059331348512160873,                            -0.0041244494998099075,                            0.10714031002520978,                            -0.12334881459178273,                            4.6859718966632009,                            -0.60635473449958388,                            1.0953543153869225                        ]                    },                    {                        \"name\": \"RightToeBase\",                        \"opposite\": \"LeftToeBase\",                        \"parent\": \"RightFoot\",                        \"zero\": [                            0.70710678052800291,                            0.0,                            0.0,                            0.70710678184509212,                            8.403361982557249e-07,                            8.4117241857978993,                            1.2258362317377616,                            -8.4033619669047652e-07                        ]                    },                    {                        \"name\": \"RightForeArm\",                        \"opposite\": \"LeftForeArm\",                        \"parent\": \"RightArm\",                        \"zero\": [                            0.98377309324262741,                            0.0,                            0.0,                            -0.17941711460123536,                            -6.1702027286257468e-13,                            14.058194418626217,                            2.5638844741830313,                            -3.3832220731925964e-12                        ]                    },                    {                        \"name\": \"LeftForeArm\",                        \"opposite\": \"RightForeArm\",                        \"parent\": \"LeftArm\",                        \"zero\": [                            0.98377309324262729,                            0.0,                            0.0,                            -0.17941711460123541,                            -1.368844213017716e-06,                            14.058198213910268,                            2.5638845604771054,                            -7.5055944835623598e-06                        ]                    },                    {                        \"name\": \"LOuterEyebrow\",                        \"opposite\": \"ROuterEyebrow\",                        \"parent\": \"Head\",                        \"zero\": [                            0.70899847856993647,                            0.18039092112945362,                            -0.1855898975441892,                            0.65600050525065901,                            -1.9776118225163026,                            7.6232343733497148,                            -0.72942612840639398,                            -0.16526300939945626                        ]                    },                    {                        \"name\": \"LeftToeBase\",                        \"opposite\": \"RightToeBase\",                        \"parent\": \"LeftFoot\",                        \"zero\": [                            0.70710678184509201,                            0.0,                            0.0,                            0.70710678052800302,                            4.3488298965582335e-06,                            8.411729670618918,                            1.2258406642075239,                            -4.3488299046585603e-06                        ]                    },                    {                        \"name\": \"RightArm\",                        \"opposite\": \"LeftArm\",                        \"parent\": \"RightShoulder\",                        \"zero\": [                            0.95902754859257711,                            -0.090009034931551976,                            -0.26698949167465702,                            -0.029684103599591026,                            0.63492256593792085,                            6.7649682085529825,                            0.20939031304207864,                            -1.8833326359365981                        ]                    },                    {                        \"name\": \"LeftArm\",                        \"opposite\": \"RightArm\",                        \"parent\": \"LeftShoulder\",                        \"zero\": [                            0.95902754555786507,                            0.090009038226455063,                            0.26698950108016006,                            -0.029684107057023023,                            -0.6349231067506913,                            6.7649733590944781,                            0.20939061194965219,                            1.8833341323863602                        ]                    },                    {                        \"name\": \"LeftUpLeg\",                        \"opposite\": \"RightUpLeg\",                        \"parent\": \"Spine\",                        \"zero\": [                            0.026459621904978258,                            -0.0018707192461622625,                            -0.99007493655950318,                            -0.13801452392716992,                            0.41416446636072496,                            4.2412346429300216,                            -0.84274874978808534,                            6.0675417852131366                        ]                    },                    {                        \"name\": \"Head\",                        \"opposite\": null,                        \"parent\": \"Neck\",                        \"zero\": [                            0.99733646984092961,                            -6.630738908823449e-14,                            2.0229921578861016e-07,                            -0.072938096528437441,                            -2.3274715669075064e-08,                            3.4540351091544763,                            0.25260205168822619,                            3.8235499005999331e-07                        ]                    },                    {                        \"name\": \"Jaw\",                        \"opposite\": null,                        \"parent\": \"Head\",                        \"zero\": [                            0.64508019954048046,                            0.0012046635476413885,                            -0.0042614046114191131,                            0.76410203858993131,                            0.00066017841210564603,                            1.5347281928034464,                            -0.85677861248328013,                            -0.0077552200650559669                        ]                    },                    {                        \"name\": \"LEyeBlinkTop\",                        \"opposite\": \"REyeBlinkTop\",                        \"parent\": \"Head\",                        \"zero\": [                            0.81371578341823614,                            0.10190217156109954,                            -0.10861938856841985,                            0.56185798888527072,                            -1.0834634034052328,                            7.3806142775246721,                            0.26967949606719266,                            0.28267475925655394                        ]                    },                    {                        \"name\": \"Spine1\",                        \"opposite\": null,                        \"parent\": \"Spine\",                        \"zero\": [                            0.99990643119760514,                            5.1897199741419777e-10,                            3.7934373214172801e-08,                            -0.01367950473034431,                            1.5487386973557272e-08,                            7.6577970101410733,                            0.098666911961088219,                            1.6961862265527472e-06                        ]                    },                    {                        \"name\": \"RightHandIndex2\",                        \"opposite\": \"LeftHandIndex2\",                        \"parent\": \"RightHandIndex1\",                        \"zero\": [                            0.99562210717940935,                            1.6690415880257212e-08,                            1.5669110477803339e-09,                            0.093469886571197328,                            -7.3561324845762322e-09,                            1.5576291817978831,                            -0.14623451174304886,                            -1.9733001623341471e-07                        ]                    },                    {                        \"name\": \"LeftHandMiddle2\",                        \"opposite\": \"RightHandMiddle2\",                        \"parent\": \"LeftHandMiddle1\",                        \"zero\": [                            0.99778029665035362,                            0.0,                            0.0,                            0.066591888517539749,                            5.7729835543519322e-08,                            1.8964867525263616,                            -0.1265734085709008,                            -8.6499562809390864e-07                        ]                    },                    {                        \"name\": \"LeftHandThumb1\",                        \"opposite\": \"RightHandThumb1\",                        \"parent\": \"LeftHand\",                        \"zero\": [                            0.66087000042611954,                            0.65393941388527854,                            0.36675118007775165,                            -0.033281487582976221,                            -1.0434188218778317,                            1.4603012312551304,                            -0.79361872506552023,                            -0.77148804967501217                        ]                    },                    {                        \"name\": \"LeftFoot\",                        \"opposite\": \"RightFoot\",                        \"parent\": \"LeftLeg\",                        \"zero\": [                            0.99872908923544557,                            0.01493641672774358,                            -0.01971790105563662,                            0.043912573919468262,                            -0.29625477914690151,                            19.809238717302822,                            -0.8709722510245389,                            -0.39110719133335536                        ]                    },                    {                        \"name\": \"LeftHandIndex1\",                        \"opposite\": \"RightHandIndex1\",                        \"parent\": \"LeftHand\",                        \"zero\": [                            0.99423517770585335,                            -0.00059332614271038488,                            0.0041244623361657851,                            0.10714032007949298,                            0.12334891447572002,                            4.6859864656753398,                            -0.60636469074629507,                            -1.095354047972626                        ]                    }                ],                \"root\": \"Root\"            }}";
		      skeletonPose.buildFromJSONDescription(JSON.parse(description)); 
		      skeletonPose.jointOrder = ['Root','Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'Jaw', 'LOuterEyebrow', 'ROuterEyebrow', 
		                          'LEyeBlinkTop', 'REyeBlinkTop', 'LMouthCorner', 'RMouthCorner', 'LeftShoulder', 'LeftArm', 
		                          'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandIndex1', 'LeftHandIndex2',
		                           'LeftHandMiddle1', 'LeftHandMiddle2', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
		                            'RightHandThumb1', 'RightHandThumb2', 'RightHandIndex1', 'RightHandIndex2', 'RightHandMiddle1',
		                             'RightHandMiddle2', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 
		                             'RightFoot', 'RightToeBase'];

		      skeletonPose.joints["Root"].setTranslation([0.0 ,0.0 ,85.0]);
		      skeletonPose.joints['Hips'].setTranslation([0, 0,0]); 
		      skeletonPose.joints['Spine'].setTranslation([15.3169, 0.0, -0.012192 ]); 
		      skeletonPose.joints['Spine1'].setTranslation([15.3176, 0.0, -0.012192 ]);
		      skeletonPose.joints['Spine2'].setTranslation([0, 0,0]); 
		      skeletonPose.joints['Neck'].setTranslation([19.914196, 0.0, 1.563385]); 
		      skeletonPose.joints['Head'].setTranslation([6.926514, 0.0, 0.0 ]); 
		      skeletonPose.joints['Jaw'].setTranslation([-0.746246, 0.0, 2.500015 ]); 
		      skeletonPose.joints['LOuterEyebrow'].setTranslation([12.54156, -4.926716, 8.292946 ]); 
		      skeletonPose.joints['ROuterEyebrow'].setTranslation([12.54156, 4.926668, 8.292969]); 
	          skeletonPose.joints['LEyeBlinkTop'].setTranslation([11.8678, -3.335859, 8.439621 ]); 
	          skeletonPose.joints['REyeBlinkTop'].setTranslation([11.86783, 3.33581, 8.439636 ]); 
	          skeletonPose.joints['LMouthCorner'].setTranslation([ 4.012383, -4.0, 9.9 ]);
	          skeletonPose.joints['RMouthCorner'].setTranslation([ 4.012383, 4.0, 10.5 ]);
	          skeletonPose.joints['LeftShoulder'].setTranslation([-5.709747, -7.695568, 0.4935]); 
	          
	          skeletonPose.joints['LeftArm'].setTranslation([14.108002, 1.5e-05, 0.0 ]); 
	          skeletonPose.joints['LeftForeArm'].setTranslation([28.5802, 0.0, 0.0 ]);
	          skeletonPose.joints['LeftHand'].setTranslation([27.091995, 0.0, 0.0 ]); 
	          skeletonPose.joints['LeftHandThumb1'].setTranslation([2.676079, 3.19825, 0.6282034 ]);
	          
	          skeletonPose.joints['LeftHandThumb2'].setTranslation([2.827835, 0.0, 3.8147e-06 ]); 
	          
	          skeletonPose.joints['LeftHandIndex1'].setTranslation([9.438965, 2.242447, -0.2039265 ]); 
	          skeletonPose.joints['LeftHandIndex2'].setTranslation([3.12896, -7.62939e-06, -9.53674e-07 ]);
	           skeletonPose.joints['LeftHandMiddle1'].setTranslation([9.559281, -0.1453552, -0.1864228 ]); 
	           skeletonPose.joints['LeftHandMiddle2'].setTranslation([3.801414, 0.0, -1.90735e-06 ]);
	           skeletonPose.joints['RightShoulder'].setTranslation([-6.751747, 6.810848, -0.282974 ]); 
	           skeletonPose.joints['RightArm'].setTranslation([4.108002, -1.5e-05, 0.0 ]); 
	           skeletonPose.joints['RightForeArm'].setTranslation([28.5802, 0.0, 0.0 ]);
	           skeletonPose.joints['RightHand'].setTranslation([27.091896, 0.0, 0.0 ]);
	            skeletonPose.joints['RightHandThumb1'].setTranslation([2.676086, -3.19825, 0.6282063 ]);
	            skeletonPose.joints['RightHandThumb2'].setTranslation([2.82782, 0.0, 0.0 ]); 
	            skeletonPose.joints['RightHandIndex1'].setTranslation([ 9.438965, -2.242447, -0.203926]); 
	            skeletonPose.joints['RightHandIndex2'].setTranslation([ 3.12896, 3.8147e-06, 3.8147e-06]); 
	            skeletonPose.joints['RightHandMiddle1'].setTranslation([9.559281, 0.1453514, -0.1864204 ]);
	            
	             skeletonPose.joints['RightHandMiddle2'].setTranslation([3.801414, 0.0, 0.0 ]);
	             skeletonPose.joints['LeftUpLeg'].setTranslation([-12.0213, -8.836849, -0.372513 ]);
	             skeletonPose.joints['LeftLeg'].setTranslation([39.9035, 0.0, 0.0 ]);
	             skeletonPose.joints['LeftFoot'].setTranslation([39.6689, 0.0, 0.0 ]); 
	             skeletonPose.joints['LeftToeBase'].setTranslation([10.162399, 0.0, 13.629623 ]); 
	             skeletonPose.joints['RightUpLeg'].setTranslation([ -12.0213, 8.836851, -0.372452 ]);
	             skeletonPose.joints['RightLeg'].setTranslation([ 39.9035, 0.0, 0.0 ]); 
	             skeletonPose.joints['RightFoot'].setTranslation([39.6689, 0.0, 0.0 ]); 
	             skeletonPose.joints['RightToeBase'].setTranslation([10.162399, -2e-06, 13.629593]); 
	             
	             
	             skeletonPose.joints["Root"].setRotation([0.704207179328, 0.0188261775827, 0.709494115937, 0.0188659235159]);
	   	      skeletonPose.joints['Hips'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	   	      skeletonPose.joints['Spine'].setRotation([0.00117388346147, 0.0266532943143, 0.00997814743926, 0.999594247919]); 
	   	      skeletonPose.joints['Spine1'].setRotation([0.0, 0.0, 0.0, 1.0]);
	   	      skeletonPose.joints['Spine2'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	   	      skeletonPose.joints['Neck'].setRotation([-0.00420721006721, -0.250859005983, -0.0646192695897, 0.96585527306]); 
	   	      skeletonPose.joints['Head'].setRotation([-0.0147440883143, 0.0114567702525, 0.00862155787308, 0.999788489139]); 
	   	      skeletonPose.joints['Jaw'].setRotation([-0.0118269251476, -0.616726649889, -0.0363076350531, 0.78625067173]); 
	   	      skeletonPose.joints['LOuterEyebrow'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	   	      skeletonPose.joints['ROuterEyebrow'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	             skeletonPose.joints['LEyeBlinkTop'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	             skeletonPose.joints['REyeBlinkTop'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	             
	             skeletonPose.joints['LMouthCorner'].setRotation([-0.311952838511, 0.653972990037, 0.340436790564, -0.599255827237]);
	             skeletonPose.joints['RMouthCorner'].setRotation([-0.0701316709982, 0.836176119727, 0.110084674585, -0.532702928417]);
	             skeletonPose.joints['LeftShoulder'].setRotation([-0.619497070471, 0.724371978329, 0.27833114918, -0.118492143562]); 
	             skeletonPose.joints['LeftArm'].setRotation([-0.475442661493, 0.0452127374525, 0.073210492682, 0.875528587635]); 
	             
	             skeletonPose.joints['LeftForeArm'].setRotation([-1.00981459091e-07, 0.29841443865, 1.35159710964e-08, 0.95443639013]);
	             skeletonPose.joints['LeftHand'].setRotation([  -0.708622390129, 0.0163307935342, -0.016251798396, 0.705211664991]); 
	             
	             skeletonPose.joints['LeftHandThumb1'].setRotation([ 0.0, 0.0, 0.0, 1.0]);
	             skeletonPose.joints['LeftHandThumb2'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	             
	             skeletonPose.joints['LeftHandIndex1'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	             skeletonPose.joints['LeftHandIndex2'].setRotation([0.0, 0.0, 0.0, 1.0]);
	              skeletonPose.joints['LeftHandMiddle1'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	              skeletonPose.joints['LeftHandMiddle2'].setRotation([0.0, 0.0, 0.0, 1.0]);
	              
	              
	              skeletonPose.joints['RightShoulder'].setRotation([ 0.556582140973, 0.788955542744, -0.213686441834, -0.148672716054]); 
	              skeletonPose.joints['RightArm'].setRotation([-0.192555991637, -0.220259486306, 0.0662360876335, 0.953950066551]); 
	              skeletonPose.joints['RightForeArm'].setRotation([3.59601555057e-08, 0.275321896414, 1.79999268675e-08, 0.961352096453]);
	              skeletonPose.joints['RightHand'].setRotation([0.781505834087, -0.152720668618, 0.464367493678, 0.387669781489]);
	              
	               skeletonPose.joints['RightHandThumb1'].setRotation([0.0, 0.0, 0.0, 1.0]);
	               skeletonPose.joints['RightHandThumb2'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	               skeletonPose.joints['RightHandIndex1'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	               skeletonPose.joints['RightHandIndex2'].setRotation([0.0, 0.0, 0.0, 1.0]); 
	               skeletonPose.joints['RightHandMiddle1'].setRotation([0.0, 0.0, 0.0, 1.0]);
	                skeletonPose.joints['RightHandMiddle2'].setRotation([0.0, 0.0, 0.0, 1.0]);
	                
	                skeletonPose.joints['LeftUpLeg'].setRotation([0.0226440010946, -0.184060420579, 0.982654062248, 6.88370978982e-05]);
	                skeletonPose.joints['LeftLeg'].setRotation([9.69563757085e-10, 0.1111038225, 8.6726178901e-09, 0.993808804864]);
	                skeletonPose.joints['LeftFoot'].setRotation([0.034572045207, -0.0555098956179, 0.0191550151436, 0.997675553762]); 
	                skeletonPose.joints['LeftToeBase'].setRotation([-2.38651075876e-07, 0.716910784084, 2.42614324521e-07, -0.697164921424]); 
	                skeletonPose.joints['RightUpLeg'].setRotation([0.0260282654649, -0.233016448498, -0.971113492885, 0.0443220945324]);
	                skeletonPose.joints['RightLeg'].setRotation([-8.71182824758e-09, 0.0582507941093, -5.08334074555e-10, 0.998301980858]); 
	                skeletonPose.joints['RightFoot'].setRotation([-0.0952900661386, -0.0442271382649, 0.0158306326511, 0.994340562688]); 
	                skeletonPose.joints['RightToeBase'].setRotation([9.41109569456e-08, -0.661966300401, -4.13833066236e-08, 0.749533599737]); 
	                console.log(skeletonPose.joints['LeftHand'].getAbsoluteTransformation());
	                console.log(skeletonPose.joints['LeftHand'].getAbsolutePosition());
		
		/*
			var joint = new SkeletonJoint();
			var joint2 = new SkeletonJoint();
			var b = [1,2,3];
			console.log(b[1]);
			joint2.setTranslation(b);
			var pose = new SkeletonPose();
			pose.addJoint("root",joint);
			pose.addJoint("test2",joint2);
			pose.joints["root"].addChild("test2");
			
			var message = {"joints" :
			{
				"root":{ "translation":
							[2,4,5]
						,
						"rotation":
								[2,4,5,1]
					},
					"test2":{ "translation":
						[2,323,5]
					,
					"rotation":
							[2,4,5,1]
					}
					
				}
			};
			pose.updateFromMessage(message);
			
			console.log(pose.translationToText());
			console.log(pose.rotationToText());
			
			var pose2 = new SkeletonPose();
			description = "		{       \"skeleton\":     {         \"joints\": [                    {                        \"name\": \"RightShoulder\",                        \"opposite\": \"LeftShoulder\",                        \"parent\": \"Neck\",                        \"zero\": [                            0.12506141884222691,                            -0.61352943311603014,                            0.11395940754115953,                            0.77133295641399879,                            0.86415573612225827,                            0.30022353452578976,                            4.6718976077279013,                            -0.59155160840700627                        ]                    },                    {                        \"name\": \"LeftLeg\",                        \"opposite\": \"RightLeg\",                        \"parent\": \"LeftUpLeg\",                        \"zero\": [                            0.99809603520891066,                            0.0,                            0.0,                            -0.06167904425534694,                            -7.610072692584324e-07,                            19.913786070358935,                            1.2306072415283222,                            -1.2314690465492345e-05                        ]                    },                    {                        \"name\": \"LeftHand\",                        \"opposite\": \"RightHand\",                        \"parent\": \"LeftForeArm\",                        \"zero\": [                            0.69832815195686082,                            -0.71476224963173718,                            0.012448761055246041,                            0.036024256190560898,                            9.6821530625171093,                            9.4595371173557528,                            -0.48798424828830811,                            0.16863012073347902                        ]                    },                    {                        \"name\": \"Spine\",                        \"opposite\": null,                        \"parent\": \"Hips\",                        \"zero\": [                            0.99975230388057579,                            -2.0642524305366003e-06,                            -6.7779596672391996e-07,                            -0.022256030211162948,                            1.2609240976387584e-05,                            6.013299514395845,                            0.052417665515597373,                            7.0821747825474009e-06                        ]                    },                    {                        \"name\": \"RightLeg\",                        \"opposite\": \"LeftLeg\",                        \"parent\": \"RightUpLeg\",                        \"zero\": [                            0.99809603498428967,                            0.0,                            0.0,                            -0.061679047890183303,                            -7.4963995031804389e-07,                            19.913782523263105,                            1.230600028086122,                            -1.2130742734719538e-05                        ]                    },                    {                        \"name\": \"LeftShoulder\",                        \"opposite\": \"RightShoulder\",                        \"parent\": \"Neck\",                        \"zero\": [                            0.12505971984034614,                            0.61352977462336244,                            -0.11396155019684859,                            0.77133264367637866,                            -0.86415826197872847,                            0.30022002821033433,                            4.6718951270622977,                            0.5915656217501406                        ]                    },                    {                        \"name\": \"REyeBlinkTop\",                        \"opposite\": \"LEyeBlinkTop\",                        \"parent\": \"Head\",                        \"zero\": [                            0.81371577820447372,                            -0.10190217247938103,                            0.10861626356906059,                            0.56185860039102953,                            1.0834636377736651,                            7.3806141932096319,                            0.26968086251784584,                            -0.2826735484324141                        ]                    },                    {                        \"name\": \"Spine2\",                        \"opposite\": null,                        \"parent\": \"Spine1\",                        \"zero\": [                            0.99987888340530762,                            6.7173601746585677e-10,                            -4.3156116663510564e-08,                            0.015563371105046339,                            -8.6907334506076807e-08,                            7.6577700413949223,                            -0.12529076485752971,                            4.9054761500601198e-06                        ]                    },                    {                        \"name\": \"RightHandMiddle2\",                        \"opposite\": \"LeftHandMiddle2\",                        \"parent\": \"RightHandMiddle1\",                        \"zero\": [                            0.99778029687837244,                            -9.9229632878896184e-10,                            1.4868083761684785e-08,                            0.06659188510101427,                            -8.7692672773952497e-08,                            1.8964851023915861,                            -0.12657213011707993,                            1.370464001329856e-06                        ]                    },                    {                        \"name\": \"RightHand\",                        \"opposite\": \"LeftHand\",                        \"parent\": \"RightForeArm\",                        \"zero\": [                            0.69832813554893403,                            0.71476226413445976,                            -0.012448768913312253,                            0.036024283791263226,                            -9.6821515645038865,                            9.4595352912864268,                            -0.48798378821553112,                            -0.16863096475856754                        ]                    },                    {                        \"name\": \"LeftHandIndex2\",                        \"opposite\": \"RightHandIndex2\",                        \"parent\": \"LeftHandIndex1\",                        \"zero\": [                            0.99562210717940935,                            -1.1126943206823083e-08,                            -1.0446072982171097e-09,                            0.093469886571197341,                            2.4263570508675945e-08,                            1.5576328684491394,                            -0.14622795672396399,                            -7.4659398657890644e-08                        ]                    },                    {                        \"name\": \"Hips\",                        \"opposite\": null,                        \"parent\": \"Root\",                        \"zero\": [                            0.50000000000000022,                            0.5,                            0.49999999999999978,                            0.5,                            -22.379499435424801,                            22.379499435424805,                            22.379499435424815,                            -22.379499435424798                        ]                    },                    {                        \"name\": \"ROuterEyebrow\",                        \"opposite\": \"LOuterEyebrow\",                        \"parent\": \"Head\",                        \"zero\": [                            0.70899847628241486,                            -0.18039093173428608,                            0.18558627043968004,                            0.65600153094450953,                            1.9776120307350906,                            7.6232345450470431,                            -0.72942682525357916,                            0.1652588673725508                        ]                    },                    {                        \"name\": \"RightHandThumb2\",                        \"opposite\": \"LeftHandThumb2\",                        \"parent\": \"RightHandThumb1\",                        \"zero\": [                            0.99363449345380406,                            0.0,                            0.0,                            0.11265209016614888,                            4.4813985956593877e-07,                            1.4049082296341495,                            -0.15927835415719879,                            -3.9527648506078586e-06                        ]                    },                    {                        \"name\": \"LeftHandMiddle1\",                        \"opposite\": \"RightHandMiddle1\",                        \"parent\": \"LeftHand\",                        \"zero\": [                            0.97923375970029547,                            -0.13920087497550512,                            -0.028465963835955347,                            0.14461690486304807,                            0.65216766382430347,                            4.6689811679674689,                            -0.79261250127106231,                            -0.077866473384806956                        ]                    },                    {                        \"name\": \"RightUpLeg\",                        \"opposite\": \"LeftUpLeg\",                        \"parent\": \"Spine\",                        \"zero\": [                            0.02645970275836412,                            0.0018708038873279089,                            0.99007531332127774,                            -0.13801180447529807,                            -0.41417600902436347,                            4.2412349941744267,                            -0.84273221770584028,                            -6.0675448947932766                        ]                    },                    {                        \"name\": \"RightHandMiddle1\",                        \"opposite\": \"LeftHandMiddle1\",                        \"parent\": \"RightHand\",                        \"zero\": [                            0.97923376656107552,                            0.13920085844323959,                            0.02846596972570907,                            0.1446168731609464,                            -0.65216641033019596,                            4.668974558551187,                            -0.79260391252170437,                            0.077863271031121881                        ]                    },                    {                        \"name\": \"LeftHandThumb2\",                        \"opposite\": \"RightHandThumb2\",                        \"parent\": \"LeftHandThumb1\",                        \"zero\": [                            0.9936344869096374,                            -2.591103659480359e-08,                            -2.9376334707265155e-09,                            0.11265214788818233,                            2.0159986875309361e-07,                            1.4049145141212764,                            -0.15927854796698954,                            -1.4591971465861785e-06                        ]                    },                    {                        \"name\": \"RMouthCorner\",                        \"opposite\": \"LMouthCorner\",                        \"parent\": \"Head\",                        \"zero\": [                            0.65090786009372759,                            -0.15812989633195329,                            0.20154573514098909,                            0.71462802226104216,                            0.23287927877434367,                            5.4361881625107999,                            2.2472349894307522,                            0.35699732118688243                        ]                    },                    {                        \"name\": \"RightHandThumb1\",                        \"opposite\": \"LeftHandThumb1\",                        \"parent\": \"RightHand\",                        \"zero\": [                            0.66086999546564162,                            -0.65393942954482009,                            -0.36675115994257024,                            -0.033281500275939985,                            1.0434149981840752,                            1.4602976753036423,                            -0.79361975359847525,                            0.77149170973945336                        ]                    },                    {                        \"name\": \"LMouthCorner\",                        \"opposite\": \"RMouthCorner\",                        \"parent\": \"Head\",                        \"zero\": [                            0.65090712485313884,                            0.15813295223489637,                            -0.20155298191915422,                            0.71462597189886223,                            -0.23286694594040869,                            5.4361863414276606,                            2.247239271880991,                            -0.35700866538928133                        ]                    },                    {                        \"name\": \"Neck\",                        \"opposite\": null,                        \"parent\": \"Spine2\",                        \"zero\": [                            0.98570937631678046,                            4.7474474331514631e-14,                            -4.6722630545441012e-07,                            0.16845481720914188,                            -9.0038947977670957e-07,                            9.9464770498855604,                            -0.90680195613947545,                            2.7534988923906539e-06                        ]                    },                    {                        \"name\": \"RightFoot\",                        \"opposite\": \"LeftFoot\",                        \"parent\": \"RightLeg\",                        \"zero\": [                            0.9987290879261993,                            -0.014936452969592489,                            0.019717985522220493,                            0.043912553441295098,                            0.29625710859441817,                            19.809244586765715,                            -0.87099161388747603,                            0.39108311774198362                        ]                    },                    {                        \"name\": \"RightHandIndex1\",                        \"opposite\": \"LeftHandIndex1\",                        \"parent\": \"RightHand\",                        \"zero\": [                            0.9942351788501218,                            0.00059331348512160873,                            -0.0041244494998099075,                            0.10714031002520978,                            -0.12334881459178273,                            4.6859718966632009,                            -0.60635473449958388,                            1.0953543153869225                        ]                    },                    {                        \"name\": \"RightToeBase\",                        \"opposite\": \"LeftToeBase\",                        \"parent\": \"RightFoot\",                        \"zero\": [                            0.70710678052800291,                            0.0,                            0.0,                            0.70710678184509212,                            8.403361982557249e-07,                            8.4117241857978993,                            1.2258362317377616,                            -8.4033619669047652e-07                        ]                    },                    {                        \"name\": \"RightForeArm\",                        \"opposite\": \"LeftForeArm\",                        \"parent\": \"RightArm\",                        \"zero\": [                            0.98377309324262741,                            0.0,                            0.0,                            -0.17941711460123536,                            -6.1702027286257468e-13,                            14.058194418626217,                            2.5638844741830313,                            -3.3832220731925964e-12                        ]                    },                    {                        \"name\": \"LeftForeArm\",                        \"opposite\": \"RightForeArm\",                        \"parent\": \"LeftArm\",                        \"zero\": [                            0.98377309324262729,                            0.0,                            0.0,                            -0.17941711460123541,                            -1.368844213017716e-06,                            14.058198213910268,                            2.5638845604771054,                            -7.5055944835623598e-06                        ]                    },                    {                        \"name\": \"LOuterEyebrow\",                        \"opposite\": \"ROuterEyebrow\",                        \"parent\": \"Head\",                        \"zero\": [                            0.70899847856993647,                            0.18039092112945362,                            -0.1855898975441892,                            0.65600050525065901,                            -1.9776118225163026,                            7.6232343733497148,                            -0.72942612840639398,                            -0.16526300939945626                        ]                    },                    {                        \"name\": \"LeftToeBase\",                        \"opposite\": \"RightToeBase\",                        \"parent\": \"LeftFoot\",                        \"zero\": [                            0.70710678184509201,                            0.0,                            0.0,                            0.70710678052800302,                            4.3488298965582335e-06,                            8.411729670618918,                            1.2258406642075239,                            -4.3488299046585603e-06                        ]                    },                    {                        \"name\": \"RightArm\",                        \"opposite\": \"LeftArm\",                        \"parent\": \"RightShoulder\",                        \"zero\": [                            0.95902754859257711,                            -0.090009034931551976,                            -0.26698949167465702,                            -0.029684103599591026,                            0.63492256593792085,                            6.7649682085529825,                            0.20939031304207864,                            -1.8833326359365981                        ]                    },                    {                        \"name\": \"LeftArm\",                        \"opposite\": \"RightArm\",                        \"parent\": \"LeftShoulder\",                        \"zero\": [                            0.95902754555786507,                            0.090009038226455063,                            0.26698950108016006,                            -0.029684107057023023,                            -0.6349231067506913,                            6.7649733590944781,                            0.20939061194965219,                            1.8833341323863602                        ]                    },                    {                        \"name\": \"LeftUpLeg\",                        \"opposite\": \"RightUpLeg\",                        \"parent\": \"Spine\",                        \"zero\": [                            0.026459621904978258,                            -0.0018707192461622625,                            -0.99007493655950318,                            -0.13801452392716992,                            0.41416446636072496,                            4.2412346429300216,                            -0.84274874978808534,                            6.0675417852131366                        ]                    },                    {                        \"name\": \"Head\",                        \"opposite\": null,                        \"parent\": \"Neck\",                        \"zero\": [                            0.99733646984092961,                            -6.630738908823449e-14,                            2.0229921578861016e-07,                            -0.072938096528437441,                            -2.3274715669075064e-08,                            3.4540351091544763,                            0.25260205168822619,                            3.8235499005999331e-07                        ]                    },                    {                        \"name\": \"Jaw\",                        \"opposite\": null,                        \"parent\": \"Head\",                        \"zero\": [                            0.64508019954048046,                            0.0012046635476413885,                            -0.0042614046114191131,                            0.76410203858993131,                            0.00066017841210564603,                            1.5347281928034464,                            -0.85677861248328013,                            -0.0077552200650559669                        ]                    },                    {                        \"name\": \"LEyeBlinkTop\",                        \"opposite\": \"REyeBlinkTop\",                        \"parent\": \"Head\",                        \"zero\": [                            0.81371578341823614,                            0.10190217156109954,                            -0.10861938856841985,                            0.56185798888527072,                            -1.0834634034052328,                            7.3806142775246721,                            0.26967949606719266,                            0.28267475925655394                        ]                    },                    {                        \"name\": \"Spine1\",                        \"opposite\": null,                        \"parent\": \"Spine\",                        \"zero\": [                            0.99990643119760514,                            5.1897199741419777e-10,                            3.7934373214172801e-08,                            -0.01367950473034431,                            1.5487386973557272e-08,                            7.6577970101410733,                            0.098666911961088219,                            1.6961862265527472e-06                        ]                    },                    {                        \"name\": \"RightHandIndex2\",                        \"opposite\": \"LeftHandIndex2\",                        \"parent\": \"RightHandIndex1\",                        \"zero\": [                            0.99562210717940935,                            1.6690415880257212e-08,                            1.5669110477803339e-09,                            0.093469886571197328,                            -7.3561324845762322e-09,                            1.5576291817978831,                            -0.14623451174304886,                            -1.9733001623341471e-07                        ]                    },                    {                        \"name\": \"LeftHandMiddle2\",                        \"opposite\": \"RightHandMiddle2\",                        \"parent\": \"LeftHandMiddle1\",                        \"zero\": [                            0.99778029665035362,                            0.0,                            0.0,                            0.066591888517539749,                            5.7729835543519322e-08,                            1.8964867525263616,                            -0.1265734085709008,                            -8.6499562809390864e-07                        ]                    },                    {                        \"name\": \"LeftHandThumb1\",                        \"opposite\": \"RightHandThumb1\",                        \"parent\": \"LeftHand\",                        \"zero\": [                            0.66087000042611954,                            0.65393941388527854,                            0.36675118007775165,                            -0.033281487582976221,                            -1.0434188218778317,                            1.4603012312551304,                            -0.79361872506552023,                            -0.77148804967501217                        ]                    },                    {                        \"name\": \"LeftFoot\",                        \"opposite\": \"RightFoot\",                        \"parent\": \"LeftLeg\",                        \"zero\": [                            0.99872908923544557,                            0.01493641672774358,                            -0.01971790105563662,                            0.043912573919468262,                            -0.29625477914690151,                            19.809238717302822,                            -0.8709722510245389,                            -0.39110719133335536                        ]                    },                    {                        \"name\": \"LeftHandIndex1\",                        \"opposite\": \"RightHandIndex1\",                        \"parent\": \"LeftHand\",                        \"zero\": [                            0.99423517770585335,                            -0.00059332614271038488,                            0.0041244623361657851,                            0.10714032007949298,                            0.12334891447572002,                            4.6859864656753398,                            -0.60636469074629507,                            -1.095354047972626                        ]                    }                ],                \"root\": \"Root\"            }}";
			pose2.buildFromJSONDescription(JSON.parse(description));
			pose2.printStructure();
			*/
	      

	 }
	//######################################################################
	//skeleton definition classes based on the Python classes using XML3D.math classes
	var SkeletonJoint = function(){
		this.name = "";//has to be set after construction
		this.parent = null;
		this.parentName = "";
		this.translation = new vec3.create();//fromValues(-1000, -1000, -1000);
		this.rotation = new quat.create();
		this.children = new Array();//array of joint names
	

	},j = SkeletonJoint.prototype;
	
	j.addChild = function(name,joint){
		this.children[name] = joint;
	};
	
	j.getTranslationString = function(){
		var translationString= "";
		translationString+=this.translation[0] +" "+this.translation[1]+" "+this.translation[2]+" ";
		return translationString
		
	};
	j.getRotationString = function(){
		var rotationString= "";
		rotationString+=this.rotation[0] +" "+this.rotation[1]+" "+this.rotation[2]+" "+this.rotation[3]+" ";
		return rotationString
	};
	j.setTranslation = function(t){
		//translation x,y,z
		this.translation[0]= t[0];
		this.translation[1]= t[1];
		this.translation[2]= t[2];
		//console.log(this.translation);
	};
	j.setRotation = function(q){
		//quaternion x,y,z,w
		this.rotation[0]= q[0];
		this.rotation[1]= q[1];
		this.rotation[2]= q[2];
		this.rotation[3]= q[3];
	};
	
	j.getRelativeTransformation =function(){
		var relativeJointTransformation = mat4.create();
        relativeJointTransformation	= mat4.fromRotationTranslation(relativeJointTransformation, this.rotation, this.translation);
        return relativeJointTransformation;
	};
	 
	
	j.getParentTransformation = function(){
    	if (this.parent != null){
        	var parentTransformation = this.parent.getAbsoluteTransformation();
        } else{
        	var parentTransformation = mat4.create();
            var parentTransformation =mat4.identity(parentTransformation);   
        }
    	return parentTransformation;
	};

    j.getAbsoluteTransformation=function(){
    	var absoluteJointTransformation = this.getParentTransformation();
        var absoluteJointTransformation = mat4.multiply( absoluteJointTransformation,this.getRelativeTransformation() ,absoluteJointTransformation) ;
        return absoluteJointTransformation;
    };
    

    j.getAbsolutePosition = function(){
    	 var absoluteParentTransformation = this.getParentTransformation();
         //var relativePosition = vec3.create();
         var relativePosition = vec3.fromValues(this.translation[0],this.translation[1],this.translation[2]);
         var absolutePosition = vec3.create();
         absolutePosition = vec3.transformMat4(absolutePosition,relativePosition,absoluteParentTransformation);
         return absolutePosition;
    };
        
	j.printStructure = function(){
		console.log(this.name);
		for (jointName in this.children){
			this.children[jointName].printStructure();
		};
	
	};
	
	
	var SkeletonPose = function(){
		this.rootJointName = "";//joint name
		this.jointOrder = new Array();
		this.joints =new Array();//hash table
		this.translationInputElement  = null;
		this.rotationInputElement = null;
		
		
	},p = SkeletonPose.prototype;
	
	p.addJoint = function(name,joint){
		this.jointOrder.push(name);
		this.joints[name] = joint;
	};
	
	p.translationToText = function(){
		var translationString = "";
		
		for (var i = 0; i< this.jointOrder.length;i++){//(jointName in this.joints)
			//console.log(this.jointOrder[i]);
			translationString += this.joints[this.jointOrder[i]].getTranslationString()
		
		}
		return  translationString;
	};
	p.rotationToText = function(){
		var rotationString = "";
		for (var i = 0; i< this.jointOrder.length;i++){//jointName in this.joints)
			rotationString += this.joints[this.jointOrder[i]].getRotationString()
		}
		return rotationString;
	};
	p.updateFromJointList = function(joints){
		for (jointName in joints){
			 if (jointName in this.joints){
				 this.joints[jointName].setTranslation(joints[jointName].translation);
				 this.joints[jointName].setRotation(joints[jointName].rotation);
			 }
		}
		this.setXflowInput()
	};
	p.setXflowInput = function(){
		
		if (this.translationInputElement  != null && this.rotationInputElement != null){
			
			this.translationInputElement.textContent = this.translationToText();
			this.rotationInputElement.textContent = this.rotationToText();
		}

	};
	
	
	p.buildFromJSONDescription= function(description){
		console.log("add joints from JSON description")
		this.joints = new Array();
		this.rootJointName = description['skeleton']['root']
		var rootJoint =new SkeletonJoint();
		rootJoint.name = this.rootJointName;
		console.log("root name "+rootJoint.name);
		this.addJoint(rootJoint.name,rootJoint);
		
		//console.log(description['skeleton']['joints']);
		
		//create dictionary of joints from the parameters
		for (index in description['skeleton']['joints']){
			 console.log(description['skeleton']['joints'][index]);
			 var joint = new SkeletonJoint();
			 joint.name = description['skeleton']['joints'][index]["name"];
			 joint.parentName = description['skeleton']['joints'][index]["parent"];
			 this.addJoint(joint.name, joint)
		}
		
		//add hierarchy information
		console.log(this.joints)
		console.log("find children");
		for (jointName in this.joints){
			console.log(jointName);
			joint = this.joints[jointName];
			console.log(joint.parentName);
			if (joint.parentName in this.joints){
				this.joints[joint.parentName].addChild(joint.name,joint);
				joint.parent = this.joints[joint.parentName];
			}
		}
	};
	p.printStructure = function(){
		if (this.rootJointName != ""){
			this.joints[this.rootJointName].printStructure();		
		}
		
	};
	


	


}());

