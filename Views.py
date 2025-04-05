from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode

from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm




# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],
        
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Close'], name="META")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Close'], name="NVDA")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JPM']['Close'], name="JPM")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')


    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df4 = yf.download(tickers = 'UBER', period='1d', interval='1d')
    df5 = yf.download(tickers = 'TSLA', period='1d', interval='1d')
    df6 = yf.download(tickers = 'X', period='1d', interval='1d')

    df1.insert(0, "Ticker", "AAPL")
    df2.insert(0, "Ticker", "AMZN")
    df3.insert(0, "Ticker", "GOOGL")
    df4.insert(0, "Ticker", "UBER")
    df5.insert(0, "Ticker", "TSLA")
    df6.insert(0, "Ticker", "X")

    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    df.reset_index(level=0, inplace=True)
    print("Columns before renaming:", df.columns)
    print("Shape of df:", df.shape)
    print(df.head())  # Print first few rows
    # Flatten the MultiIndex column names
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    print("Columns after flattening:", df.columns)  # Debug print
    df.rename(columns={'Date_': 'Date', 'Ticker_': 'Ticker'}, inplace=True)
    # Dynamically keep only required stock columns
    stock_columns = ['Date', 'Ticker'] + [col for col in df.columns if col.startswith(('Open_', 'High_', 'Low_', 'Close_', 'Volume_'))]
    df = df[stock_columns]

    # Convert 'Date' column to string if needed
    df['Date'] = df['Date'].astype(str)

    # Now rename the columns properly
    df.columns = ['Date', 'Ticker', 'Open_AAPL', 'High_AAPL', 'Low_AAPL', 'Close_AAPL', 'Volume_AAPL',
              'Close_AMZN', 'High_AMZN', 'Low_AMZN', 'Open_AMZN', 'Volume_AMZN',
              'Close_GOOGL', 'High_GOOGL', 'Low_GOOGL', 'Open_GOOGL', 'Volume_GOOGL',
              'Close_UBER', 'High_UBER', 'Low_UBER', 'Open_UBER', 'Volume_UBER',
              'Close_TSLA', 'High_TSLA', 'Low_TSLA', 'Open_TSLA', 'Volume_TSLA',
              'Close_X', 'High_X', 'Low_X', 'Open_X', 'Volume_X']

    print("Final Columns:", df.columns)  # Debug print

    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)
    print(recent_stocks)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks,
        'data': df.to_html()
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    Valid_Ticker = [
    "A","AA","AAC","AACG","AACIW","AADI","AAIC","AAIN","AAL","AAMC","AAME","AAN","AAOI","AAON","AAP","AAPL","AAQC","AAT","AATC","AAU","AAWW","AB","ABB","ABBV","ABC","ABCB","ABCL","ABCM","ABEO","ABEV","ABG","ABIO","ABM","ABMD","ABNB","ABOS","ABR","ABSI","ABST","ABT","ABTX","ABUS","ABVC","AC","ACA","ACAB","ACAD","ACAQ","ACAXR","ACB","ACBA","ACC","ACCD","ACCO","ACEL","ACER","ACET","ACEV","ACEVW","ACGL","ACGLN","ACGLO","ACH","ACHC","ACHL","ACHR","ACHV","ACI","ACII","ACIU","ACIW","ACKIT","ACLS","ACLX","ACM","ACMR","ACN","ACNB","ACON","ACOR","ACP","ACQR","ACQRU","ACR","ACRE","ACRS","ACRX","ACST","ACT","ACTD","ACTDW","ACTG","ACU","ACV","ACVA","ACXP","ADAG","ADALW","ADAP","ADBE","ADC","ADCT","ADER","ADES","ADEX","ADGI","ADI","ADIL","ADM","ADMA","ADMP","ADN","ADNT","ADNWW","ADP","ADPT","ADRA","ADRT","ADSE","ADSEW","ADSK","ADT","ADTH","ADTN","ADTX","ADUS","ADV","ADVM","ADX","ADXN","AE","AEAC","AEACW","AEAE","AEAEW","AEE","AEF","AEFC","AEG","AEHAW","AEHL","AEHR","AEI","AEIS","AEL","AEM","AEMD","AENZ","AEO","AEP","AEPPZ","AER","AERC","AERI","AES","AESC","AESE","AEVA","AEY","AEYE","AEZS","AFAQ","AFAR","AFB","AFBI","AFCG","AFG","AFGB","AFGC","AFGD","AFGE","AFIB","AFL","AFMD","AFRI","AFRIW","AFRM","AFT","AFTR","AFYA","AG","AGAC","AGBAR","AGCB","AGCO","AGD","AGE","AGEN","AGFS","AGFY","AGGR","AGI","AGIL","AGILW","AGIO","AGL","AGLE","AGM","AGMH","AGNC","AGNCM","AGNCN","AGNCO","AGNCP","AGO","AGR","AGRI","AGRO","AGRX","AGS","AGTC","AGTI","AGX","AGYS","AHCO","AHG","AHH","AHI","AHPA","AHPI","AHRNW","AHT","AI","AIB","AIC","AIF","AIG","AIH","AIHS","AIKI","AIM","AIMAW","AIMC","AIN","AINC","AINV","AIO","AIP","AIR","AIRC","AIRG","AIRI","AIRS","AIRT","AIRTP","AIT","AIU","AIV","AIZ","AIZN","AJG","AJRD","AJX","AJXA","AKA","AKAM","AKAN","AKBA","AKICU","AKR","AKRO","AKTS","AKTX","AKU","AKUS","AKYA","AL","ALB","ALBO","ALC","ALCC","ALCO","ALDX","ALE","ALEC","ALEX","ALF","ALFIW","ALG","ALGM","ALGN","ALGS","ALGT","ALHC","ALIM","ALIT","ALJJ","ALK","ALKS","ALKT","ALL","ALLE","ALLG","ALLK","ALLO","ALLR","ALLT","ALLY","ALNA","ALNY","ALORW","ALOT","ALPA","ALPN","ALPP","ALR","ALRM","ALRN","ALRS","ALSA","ALSAR","ALSAU","ALSAW","ALSN","ALT","ALTG","ALTO","ALTR","ALTU","ALTUU","ALTUW","ALV","ALVO","ALVR","ALX","ALXO","ALYA","ALZN","AM","AMAL","AMAM","AMAO","AMAOW","AMAT","AMBA","AMBC","AMBO","AMBP","AMC","AMCI","AMCR","AMCX","AMD","AME","AMED","AMEH","AMG","AMGN","AMH","AMK","AMKR","AMLX","AMN","AMNB","AMOT","AMOV","AMP","AMPE","AMPG","AMPH","AMPI","AMPL","AMPS","AMPY","AMR","AMRC","AMRK","AMRN","AMRS","AMRX","AMS","AMSC","AMSF","AMST","AMSWA","AMT","AMTB","AMTD","AMTI","AMTX","AMWD","AMWL","AMX","AMYT","AMZN","AN","ANAB","ANAC","ANDE","ANEB","ANET","ANF","ANGH","ANGHW","ANGI","ANGN","ANGO","ANIK","ANIP","ANIX","ANNX","ANPC","ANSS","ANTE","ANTX","ANVS","ANY","ANZU","ANZUW","AOD","AOGO","AOMR","AON","AORT","AOS","AOSL","AOUT","AP","APA","APAC","APACW","APAM","APCX","APD","APDN","APEI","APEN","APG","APGB","APH","API","APLD","APLE","APLS","APLT","APM","APMIU","APO","APOG","APP","APPF","APPH","APPHW","APPN","APPS","APRE","APRN","APT","APTM","APTO","APTV","APTX","APVO","APWC","APXI","APYX","AQB","AQMS","AQN","AQNA","AQNB","AQNU","AQST","AQUA","AR","ARAV","ARAY","ARBE","ARBEW","ARBK","ARBKL","ARC","ARCB","ARCC","ARCE","ARCH","ARCK","ARCKW","ARCO","ARCT","ARDC","ARDS","ARDX","ARE","AREB","AREC","AREN","ARES","ARGD","ARGO","ARGU","ARGUU","ARGUW","ARGX","ARHS","ARI","ARIS","ARIZW","ARKO","ARKOW","ARKR","ARL","ARLO","ARLP","ARMK","ARMP","ARNC","AROC","AROW","ARQQ","ARQQW","ARQT","ARR","ARRWU","ARRWW","ARRY","ARTE","ARTEW","ARTL","ARTNA","ARTW","ARVL","ARVN","ARW","ARWR","ASA","ASAI","ASAN","ASAQ","ASAX","ASAXU","ASB","ASC","ASCAU","ASCB","ASCBR","ASG","ASGI","ASGN","ASH","ASIX","ASLE","ASLN","ASM","ASMB","ASML","ASND","ASNS","ASO","ASPA","ASPC","ASPCU","ASPCW","ASPN","ASPS","ASPU","ASR","ASRT","ASRV","ASTC","ASTE","ASTL","ASTLW","ASTR","ASTS","ASTSW","ASUR","ASX","ASXC","ASYS","ASZ","ATA","ATAI","ATAQ","ATAX","ATC","ATCO","ATCX","ATEC","ATEN","ATER","ATEX","ATGE","ATHA","ATHE","ATHM","ATHX","ATI","ATIF","ATIP","ATKR","ATLC","ATLCL","ATLCP","ATLO","ATNF","ATNFW","ATNI","ATNM","ATNX","ATO","ATOM","ATOS","ATR","ATRA","ATRC","ATRI","ATRO","ATSG","ATTO","ATUS","ATVC","ATVCU","ATVI","ATXI","ATXS","ATY","AU","AUB","AUBAP","AUBN","AUD","AUDC","AUGX","AUID","AUMN","AUPH","AUR","AURA","AURC","AUROW","AUS","AUST","AUTL","AUTO","AUUD","AUVI","AUY","AVA","AVAC","AVAH","AVAL","AVAN","AVAV","AVB","AVCO","AVCT","AVCTW","AVD","AVDL","AVDX","AVEO","AVGO","AVGOP","AVGR","AVID","AVIR","AVK","AVLR","AVNS","AVNT","AVNW","AVO","AVPT","AVPTW","AVRO","AVT","AVTE","AVTR","AVTX","AVXL","AVY","AVYA","AWF","AWH","AWI","AWK","AWP","AWR","AWRE","AWX","AX","AXAC","AXDX","AXGN","AXL","AXLA","AXNX","AXON","AXP","AXR","AXS","AXSM","AXTA","AXTI","AXU","AY","AYI","AYLA","AYRO","AYTU","AYX","AZ","AZEK","AZN","AZO","AZPN","AZRE","AZTA","AZUL","AZYO","AZZ","B","BA","BABA","BAC","BACA","BAFN","BAH","BAK","BALL","BALY","BAM","BAMH","BAMI","BAMR","BANC","BAND","BANF","BANFP","BANR","BANX","BAOS","BAP","BARK","BASE","BATL","BATRA","BATRK","BAX","BB","BBAI","BBAR","BBBY","BBCP","BBD","BBDC","BBDO","BBGI","BBI","BBIG","BBIO","BBLG","BBLN","BBN","BBQ","BBSI","BBU","BBUC","BBVA","BBW","BBWI","BBY","BC","BCAB","BCAC","BCACU","BCACW","BCAN","BCAT","BCBP","BCC","BCDA","BCDAW","BCE","BCEL","BCH","BCLI","BCML","BCO","BCOR","BCOV","BCOW","BCPC","BCRX","BCS","BCSA","BCSAW","BCSF","BCTX","BCTXW","BCV","BCX","BCYC","BDC","BDJ","BDL","BDN","BDSX","BDTX","BDX","BDXB","BE","BEAM","BEAT","BECN","BEDU","BEEM","BEKE","BELFA","BELFB","BEN","BENE","BENER","BENEW","BEP","BEPC","BEPH","BEPI","BERY","BEST","BFAC","BFAM","BFC","BFH","BFI","BFIIW","BFIN","BFK","BFLY","BFRI","BFRIW","BFS","BFST","BFZ","BG","BGB","BGCP","BGFV","BGH","BGI","BGNE","BGR","BGRY","BGRYW","BGS","BGSF","BGSX","BGT","BGX","BGXX","BGY","BH","BHAC","BHACU","BHAT","BHB","BHC","BHE","BHF","BHFAL","BHFAM","BHFAN","BHFAO","BHFAP","BHG","BHIL","BHK","BHLB","BHP","BHR","BHSE","BHSEW","BHV","BHVN","BIDU","BIG","BIGC","BIGZ","BIIB","BILI","BILL","BIMI","BIO","BIOC","BIOL","BIOR","BIOSW","BIOT","BIOTU","BIOTW","BIOX","BIP","BIPC","BIPH","BIPI","BIRD","BIT","BITF","BIVI","BJ","BJDX","BJRI","BK","BKCC","BKD","BKE","BKEP","BKEPP","BKH","BKI","BKKT","BKN","BKNG","BKR","BKSC","BKSY","BKT","BKTI","BKU","BKYI","BL","BLBD","BLBX","BLCM","BLCO","BLCT","BLD","BLDE","BLDEW","BLDP","BLDR","BLE","BLEU","BLEUU","BLEUW","BLFS","BLFY","BLI","BLIN","BLK","BLKB","BLMN","BLND","BLNG","BLNGW","BLNK","BLNKW","BLPH","BLRX","BLSA","BLTE","BLTS","BLTSW","BLU","BLUA","BLUE","BLW","BLX","BLZE","BMA","BMAC","BMAQ","BMAQR","BMBL","BME","BMEA","BMEZ","BMI","BMO","BMRA","BMRC","BMRN","BMTX","BMY","BNED","BNFT","BNGO","BNL","BNOX","BNR","BNRG","BNS","BNSO","BNTC","BNTX","BNY","BOAC","BOAS","BOC","BODY","BOE","BOH","BOKF","BOLT","BON","BOOM","BOOT","BORR","BOSC","BOTJ","BOWL","BOX","BOXD","BOXL","BP","BPAC","BPMC","BPOP","BPOPM","BPRN","BPT","BPTH","BPTS","BPYPM","BPYPN","BPYPO","BPYPP","BQ","BR","BRAC","BRACR","BRAG","BRBR","BRBS","BRC","BRCC","BRCN","BRDG","BRDS","BREZ","BREZR","BREZW","BRFH","BRFS","BRG","BRID","BRIV","BRIVW","BRKHU","BRKL","BRKR","BRLI","BRLT","BRMK","BRN","BRO","BROG","BROS","BRP","BRPM","BRPMU","BRPMW","BRQS","BRSP","BRT","BRTX","BRW","BRX","BRY","BRZE","BSAC","BSBK","BSBR","BSET","BSFC","BSGA","BSGAR","BSGM","BSIG","BSKY","BSKYW","BSL","BSM","BSMX","BSQR","BSRR","BST","BSTZ","BSVN","BSX","BSY","BTA","BTAI","BTB","BTBD","BTBT","BTCM","BTCS","BTCY","BTG","BTI","BTMD","BTMDW","BTN","BTO","BTOG","BTRS","BTT","BTTR","BTTX","BTU","BTWN","BTWNU","BTWNW","BTX","BTZ","BUD","BUI","BUR","BURL","BUSE","BV","BVH","BVN","BVS","BVXV","BW","BWA","BWAC","BWACW","BWAQR","BWAY","BWB","BWC","BWCAU","BWEN","BWFG","BWG","BWMN","BWMX","BWNB","BWSN","BWV","BWXT","BX","BXC","BXMT","BXMX","BXP","BXRX","BXSL","BY","BYD","BYFC","BYM","BYN","BYND","BYRN","BYSI","BYTS","BYTSW","BZ","BZFD","BZFDW","BZH","BZUN","C","CAAP","CAAS","CABA","CABO","CAC","CACC","CACI","CADE","CADL","CAE","CAF","CAG","CAH","CAJ","CAKE","CAL","CALA","CALB","CALM","CALT","CALX","CAMP","CAMT","CAN","CANF","CANG","CANO","CAPD","CAPL","CAPR","CAR","CARA","CARE","CARG","CARR","CARS","CARV","CASA","CASH","CASI","CASS","CASY","CAT","CATC","CATO","CATY","CB","CBAN","CBAT","CBAY","CBD","CBFV","CBH","CBIO","CBL","CBNK","CBOE","CBRE","CBRG","CBRL","CBSH","CBT","CBTX","CBU","CBZ","CC","CCAP","CCB","CCBG","CCCC","CCCS","CCD","CCEL","CCEP","CCF","CCI","CCJ","CCK","CCL","CCLP","CCM","CCNC","CCNE","CCNEP","CCO","CCOI","CCRD","CCRN","CCS","CCSI","CCU","CCV","CCVI","CCXI","CCZ","CD","CDAK","CDAY","CDE","CDEV","CDLX","CDMO","CDNA","CDNS","CDR","CDRE","CDRO","CDTX","CDW","CDXC","CDXS","CDZI","CDZIP","CE","CEA","CEAD","CEADW","CECE","CEE","CEG","CEI","CEIX","CELC","CELH","CELU","CELZ","CEM","CEMI","CEN","CENN","CENQW","CENT","CENTA","CENX","CEPU","CEQP","CERE","CERS","CERT","CET","CETX","CETXP","CEV","CEVA","CF","CFB","CFBK","CFFE","CFFI","CFFN","CFG","CFIV","CFIVW","CFLT","CFMS","CFR","CFRX","CFSB","CFVI","CFVIW","CG","CGA","CGABL","CGAU","CGBD","CGC","CGEM","CGEN","CGNT","CGNX","CGO","CGRN","CGTX","CHAA","CHCI","CHCO","CHCT","CHD","CHDN","CHE","CHEA","CHEF","CHEK","CHGG","CHH","CHI","CHK","CHKEL","CHKEW","CHKEZ","CHKP","CHMG","CHMI","CHN","CHNG","CHNR","CHPT","CHRA","CHRB","CHRD","CHRS","CHRW","CHS","CHSCL","CHSCM","CHSCN","CHSCO","CHSCP","CHT","CHTR","CHUY","CHW","CHWA","CHWAW","CHWY","CHX","CHY","CI","CIA","CIB","CIDM","CIEN","CIF","CIFR","CIFRW","CIG","CIGI","CIH","CII","CIIGW","CIK","CIM","CINC","CINF","CING","CINT","CIO","CION","CIR","CISO","CITEW","CIVB","CIVI","CIX","CIXX","CIZN","CJJD","CKPT","CKX","CL","CLAQW","CLAR","CLAS","CLAYU","CLB","CLBK","CLBS","CLBT","CLBTW","CLDT","CLDX","CLEU","CLF","CLFD","CLGN","CLH","CLIM","CLIR","CLLS","CLM","CLMT","CLNE","CLNN","CLOV","CLPR","CLPS","CLPT","CLR","CLRB","CLRO","CLS","CLSD","CLSK","CLSN","CLST","CLVR","CLVRW","CLVS","CLVT","CLW","CLWT","CLX","CLXT","CM","CMA","CMAX","CMAXW","CMBM","CMC","CMCA","CMCL","CMCM","CMCO","CMCSA","CMCT","CME","CMG","CMI","CMLS","CMMB","CMP","CMPO","CMPOW","CMPR","CMPS","CMPX","CMRA","CMRAW","CMRE","CMRX","CMS","CMSA","CMSC","CMSD","CMT","CMTG","CMTL","CMU","CNA","CNC","CNCE","CND","CNDB","CNDT","CNET","CNEY","CNF","CNFRL","CNHI","CNI","CNK","CNM","CNMD","CNNB","CNNE","CNO","CNOB","CNOBP","CNP","CNQ","CNR","CNS","CNSL","CNSP","CNTA","CNTB","CNTG","CNTQ","CNTQW","CNTX","CNTY","CNVY","CNX","CNXA","CNXC","CNXN","CO","COCO","COCP","CODA","CODI","CODX","COE","COF","COFS","COGT","COHN","COHU","COIN","COKE","COLB","COLD","COLI","COLIU","COLIW","COLL","COLM","COMM","COMP","COMS","COMSP","COMSW","CONN","CONX","CONXW","COO","COOK","COOL","COOLU","COOP","COP","CORR","CORS","CORT","CORZ","CORZW","COSM","COST","COTY","COUP","COUR","COVA","COVAU","COVAW","COWN","COWNL","CP","CPA","CPAC","CPAR","CPARU","CPARW","CPB","CPE","CPF","CPG","CPHC","CPHI","CPIX","CPK","CPLP","CPNG","CPOP","CPRI","CPRT","CPRX","CPS","CPSH","CPSI","CPSS","CPT","CPTK","CPTN","CPTNW","CPUH","CPZ","CQP","CR","CRAI","CRBP","CRBU","CRC","CRCT","CRDF","CRDL","CRDO","CREC","CREG","CRESW","CRESY","CREX","CRF","CRGE","CRGY","CRH","CRHC","CRI","CRIS","CRK","CRKN","CRL","CRM","CRMD","CRMT","CRNC","CRNT","CRNX","CRON","CROX","CRS","CRSP","CRSR","CRT","CRTD","CRTDW","CRTO","CRTX","CRU","CRUS","CRVL","CRVS","CRWD","CRWS","CRXT","CRXTW","CS","CSAN","CSBR","CSCO","CSCW","CSGP","CSGS","CSII","CSIQ","CSL","CSPI","CSQ","CSR","CSSE","CSSEN","CSSEP","CSTE","CSTL","CSTM","CSTR","CSV","CSWC","CSWI","CSX","CTAQ","CTAS","CTBB","CTBI","CTDD","CTEK","CTG","CTGO","CTHR","CTIB","CTIC","CTKB","CTLP","CTLT","CTMX","CTO","CTOS","CTR","CTRA","CTRE","CTRM","CTRN","CTS","CTSH","CTSO","CTT","CTV","CTVA","CTXR","CTXRW","CTXS","CUBA","CUBE","CUBI","CUE","CUEN","CUK","CULL","CULP","CURI","CURO","CURV","CUTR","CUZ","CVAC","CVBF","CVCO","CVCY","CVE","CVEO","CVET","CVGI","CVGW","CVI","CVII","CVLG","CVLT","CVLY","CVM","CVNA","CVR","CVRX","CVS","CVT","CVV","CVX","CW","CWAN","CWBC","CWBR","CWCO","CWEN","CWH","CWK","CWST","CWT","CX","CXAC","CXDO","CXE","CXH","CXM","CXW","CYAN","CYBE","CYBN","CYBR","CYCC","CYCCP","CYCN","CYD","CYH","CYN","CYRN","CYRX","CYT","CYTH","CYTK","CYTO","CYXT","CZNC","CZOO","CZR","CZWI","D","DAC","DADA","DAIO","DAKT","DAL","DALN","DAN","DAO","DAOO","DAOOU","DAOOW","DAR","DARE","DASH","DATS","DAVA","DAVE","DAVEW","DAWN","DB","DBD","DBGI","DBI","DBL","DBRG","DBTX","DBVT","DBX","DC","DCBO","DCF","DCFC","DCFCW","DCGO","DCGOW","DCI","DCO","DCOM","DCOMP","DCP","DCPH","DCRD","DCRDW","DCT","DCTH","DD","DDD","DDF","DDI","DDL","DDOG","DDS","DDT","DE","DEA","DECA","DECK","DEI","DELL","DEN","DENN","DEO","DESP","DEX","DFFN","DFH","DFIN","DFP","DFS","DG","DGHI","DGICA","DGII","DGLY","DGNU","DGX","DH","DHACW","DHBC","DHBCU","DHC","DHCAU","DHCNI","DHCNL","DHF","DHHC","DHI","DHIL","DHR","DHT","DHX","DHY","DIAX","DIBS","DICE","DIN","DINO","DIOD","DIS","DISA","DISH","DIT","DK","DKL","DKNG","DKS","DLA","DLB","DLCA","DLHC","DLNG","DLO","DLPN","DLR","DLTH","DLTR","DLX","DLY","DM","DMA","DMAC","DMB","DMF","DMLP","DMO","DMRC","DMS","DMTK","DNA","DNAA","DNAB","DNAC","DNAD","DNAY","DNB","DNLI","DNMR","DNN","DNOW","DNP","DNUT","DNZ","DO","DOC","DOCN","DOCS","DOCU","DOGZ","DOLE","DOMA","DOMO","DOOO","DOOR","DORM","DOUG","DOV","DOW","DOX","DOYU","DPG","DPRO","DPSI","DPZ","DQ","DRCT","DRD","DRE","DRH","DRI","DRIO","DRMA","DRMAW","DRQ","DRRX","DRTS","DRTSW","DRTT","DRUG","DRVN","DS","DSAC","DSACU","DSACW","DSEY","DSGN","DSGR","DSGX","DSKE","DSL","DSM","DSP","DSS","DSU","DSWL","DSX","DT","DTB","DTC","DTE","DTEA","DTF","DTG","DTIL","DTM","DTOCW","DTP","DTSS","DTST","DTW","DUK","DUKB","DUNE","DUNEW","DUO","DUOL","DUOT","DV","DVA","DVAX","DVN","DWAC","DWACU","DWACW","DWIN","DWSN","DX","DXC","DXCM","DXF","DXLG","DXPE","DXR","DXYN","DY","DYAI","DYFN","DYN","DYNT","DZSI","E","EA","EAC","EACPW","EAD","EAF","EAI","EAR","EARN","EAST","EAT","EB","EBACU","EBAY","EBC","EBET","EBF","EBIX","EBMT","EBON","EBR","EBS","EBTC","EC","ECAT","ECC","ECCC","ECCW","ECCX","ECF","ECL","ECOM","ECOR","ECPG","ECVT","ED","EDAP","EDBL","EDBLW","EDD","EDF","EDI","EDIT","EDN","EDNC","EDR","EDRY","EDSA","EDTK","EDTX","EDU","EDUC","EE","EEA","EEFT","EEIQ","EEX","EFC","EFL","EFOI","EFR","EFSC","EFSCP","EFT","EFTR","EFX","EGAN","EGB
