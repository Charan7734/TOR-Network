import joblib
import numpy as np
import sm
from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
from scipy.stats import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lazypredict.Supervised import  LazyClassifier
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import seaborn as sns

plt.switch_backend('Agg')

app=Flask(__name__)


scene_b_arr=[]
@app.route('/',methods=['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def pred():
    return render_template('inputs.html')


@app.route('/submit',methods=['POST'])
def submission():
    if request.method=="POST":
        source_port = request.form['source_port']
        destination_port = request.form['destination_port']
        protocol = request.form['protocol']
        flow_duration = request.form['flow_duration']
        flow_bytes = request.form['flow_bytes']
        flow_packets = request.form['flow_packets']
        flow_iat_mean = request.form['flow_iat_mean']
        flow_iat_std = request.form['flow_iat_std']
        flow_iat_max = request.form['flow_iat_max']
        flow_iat_min = request.form['flow_iat_min']
        fwd_iat_mean = request.form['fwd_iat_mean']
        fwd_iat_std = request.form['fwd_iat_std']
        fwd_iat_max = request.form['fwd_iat_max']
        fwd_iat_min = request.form['fwd_iat_min']
        bwd_iat_mean = request.form['bwd_iat_mean']
        bwd_iat_std = request.form['bwd_iat_std']
        bwd_iat_max = request.form['bwd_iat_max']
        bwd_iat_min = request.form['bwd_iat_min']
        active_mean = request.form['active_mean']
        active_std = request.form['active_std']
        active_max = request.form['active_max']
        active_min = request.form['active_min']
        idle_mean = request.form['idle_mean']
        idle_std = request.form['idle_std']
        idle_max = request.form['idle_max']
        idle_min = request.form['idle_min']

        arr=np.array([source_port,destination_port,protocol,flow_duration,flow_bytes,
                      flow_packets,flow_iat_mean,flow_iat_std,flow_iat_max,flow_iat_min,
                      fwd_iat_std,fwd_iat_max,fwd_iat_min,fwd_iat_mean,bwd_iat_mean,
                      bwd_iat_std,bwd_iat_max,bwd_iat_min,active_mean,active_std,active_max,
                      active_min,idle_mean,idle_std,idle_max,idle_min])

        arr1 = np.array([source_port, destination_port, protocol, flow_duration,
                        flow_iat_std, flow_iat_max, flow_iat_min, fwd_iat_mean,
                        fwd_iat_std, fwd_iat_max, fwd_iat_min, bwd_iat_mean,
                        bwd_iat_std, bwd_iat_max, bwd_iat_min, active_mean, active_std, active_max,
                        active_min, idle_mean, idle_std, idle_max, idle_min])
        print(arr1)


        scene_b_arr.append(arr1)
        model=joblib.load('rf_tor.pkl')
        m2=joblib.load('random_forest_b.pkl')
        prediction1=m2.predict([arr1])
        predictions=model.predict([arr])
        if predictions[0]==0:
            result="Its TOR communication"
            if prediction1[0] == 0:
                result_b = result+"Type of communication is "+"AUDIO"
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 1:
                result_b = result+"Type of communication is "+result
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 2:
                result_b = result+"Type of communication is "+"CHAT"
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 3:
                result_b = result+"Type of communication is "+"FILE-TRANSFER"
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 4:
                result_b = result+"Type of communication is "+"MAIL"
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 5:
                result_b = result+"Type of communication is "+"P2P"
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 6:
                result_b = result+"Type of communication is "+"VIDEO"
                return render_template('inputs.html', result_b=result_b)
            elif prediction1[0] == 7:
                result_b = result+"Type of communication is "+"VOIP"
                return render_template('inputs.html', result_b=result_b)
        elif predictions[0]==1:
            result="Its not a Tor Communication"
            return render_template('inputs.html',result=result)

'''def prediction_b():
    model=joblib.load('random_forest_b.pkl')
    predictions=model.predict(scene_b_arr)
    print(predictions)
    if predictions[0]==0:
        result_b="AUDIO"
        return render_template('inputs.html',result_b=result_b)
    elif predictions[0]==1:
        result_b="BROWSING"
        return render_template('inputs.html', result_b=result_b)
    elif predictions[0]==2:
        result_b="CHAT"
        return render_template('inputs.html', result_b=result_b)
    elif predictions[0] ==3:
        result_b = "FILE-TRANSFER"
        return render_template('inputs.html', result_b=result_b)
    elif predictions[0] ==4:
        result_b = "MAIL"
        return render_template('inputs.html', result_b=result_b)
    elif predictions[0] ==5:
        result_b = "P2P"
        return render_template('inputs.html', result_b=result_b)
    elif predictions[0] ==6:
        result_b = "VIDEO"
        return render_template('inputs.html', result_b=result_b)
    elif predictions[0] ==7:
        result_b = "VOIP"
        return render_template('inputs.html', result_b=result_b)
'''

@app.route('/xai',methods=['GET',"POST"])
def xai():
    return render_template('xai.html')

@app.route('/xai_scene_a',methods=['GET','POST'])
def scene_a():
    return render_template('lime_explanation_scene_a.html')

@app.route('/xai_scene_b',methods=['GET','POST'])
def scene_b():
    return render_template('lime_explanation_scene_b.html')

@app.route('/automl',methods=['GET',"POST"])
def automl():
    return render_template('automl.html')

@app.route('/auto_scene_b',methods=['GET','POST'])
def auto():
    data = pd.read_csv('Scenario-B-merged_5s.csv')
    data.columns = data.columns.str.strip()
    print(data.columns)
    print(data.info())
    print(data.isna().sum())
    data = data.replace([-np.inf, np.inf], np.NAN)
    print(data.info())

    def ip_convert(ip):
        ip = str(ip).split('.')
        if ip[0] < '127' and ip[0] > '0':
            return 'A'
        elif ip[0] > '128' and ip[0] < '192':
            return 'B'
        elif ip[0] > '192' and ip[0] < '223':
            return 'C'
        elif ip[0] > '223' and ip[0] < '240':
            return 'D'
        elif ip[0] > '240':
            return 'E'

    data['Source IP'] = data['Source IP'].apply(ip_convert)
    data['Destination IP'] = data['Destination IP'].apply(ip_convert)

    lab = LabelEncoder()

    for i in data.select_dtypes(include='object').columns.values:
        data[i] = lab.fit_transform(data[i])
    print(data.label.value_counts())

    x = []
    for i in data.columns.values:
        data['z-scores'] = (data[i] - data[i].mean()) / data[i].std()
        outliers = np.abs(data['z-scores'] > 3).sum()
        if outliers > 0:
            x.append(i)

    print(len(data))
    thresh = 3
    for i in x[:5]:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]
    print(len(data))

    x = data.drop(['label', 'z-scores', 'Flow Bytes/s', 'Flow Packets/s'], axis=1)
    y = data.label

    def backward_ele(x, y, val=0.03):
        for i in range(0, x.shape[1]):
            linear = sm.OLS(y, x).fit()
            if max(linear.pvalues) > val:
                index = np.argmax(linear.pvalues)
                column = x.columns[index]
                x = x.drop(columns=[column])
            else:
                break
        return x

    back = backward_ele(x, y)
    back = back.drop(['Source IP', 'Destination IP'], axis=1)
    print('--------------------------------')
    print(len(back.columns.values))
    print('-------------------------------')

    x_train, x_test, y_train, y_test = train_test_split(back, y, train_size=0.7, test_size=0.3, random_state=20)

    print(back.columns.values)

    lazy = LazyClassifier()
    models, prdiction = lazy.fit(x_train, x_test, y_train, y_test)
    models_html=models.to_html()
    return render_template('auto_scene_b.html',models=models_html)

@app.route('/report',methods=['GET',"POST"])
def report():
    return render_template('report.html')


@app.route('/train_models_scene_b', methods=['GET', 'POST'])
def train_models():
    data = pd.read_csv('Scenario-B-merged_5s.csv')
    data.columns = data.columns.str.strip()
    print(data.columns)
    print(data.info())
    print(data.isna().sum())
    data = data.replace([-np.inf, np.inf], np.NAN)
    print(data.info())

    def ip_convert(ip):
        ip = str(ip).split('.')
        if ip[0] < '127' and ip[0] > '0':
            return 'A'
        elif ip[0] > '128' and ip[0] < '192':
            return 'B'
        elif ip[0] > '192' and ip[0] < '223':
            return 'C'
        elif ip[0] > '223' and ip[0] < '240':
            return 'D'
        elif ip[0] > '240':
            return 'E'

    data['Source IP'] = data['Source IP'].apply(ip_convert)
    data['Destination IP'] = data['Destination IP'].apply(ip_convert)

    lab = LabelEncoder()

    for i in data.select_dtypes(include='object').columns.values:
        data[i] = lab.fit_transform(data[i])
    print(data.label.value_counts())

    x = []
    for i in data.columns.values:
        data['z-scores'] = (data[i] - data[i].mean()) / data[i].std()
        outliers = np.abs(data['z-scores'] > 3).sum()
        if outliers > 0:
            x.append(i)

    print(len(data))
    thresh = 3
    for i in x[:5]:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]
    print(len(data))

    x = data.drop(['label', 'z-scores', 'Flow Bytes/s', 'Flow Packets/s'], axis=1)
    y = data.label

    def backward_ele(x, y, val=0.03):
        for i in range(0, x.shape[1]):
            linear = sm.OLS(y, x).fit()
            if max(linear.pvalues) > val:
                index = np.argmax(linear.pvalues)
                column = x.columns[index]
                x = x.drop(columns=[column])
            else:
                break
        return x

    back = backward_ele(x, y)
    back = back.drop(['Source IP', 'Destination IP'], axis=1)
    print('--------------------------------')
    print(len(back.columns.values))
    print('-------------------------------')

    x_train, x_test, y_train, y_test = train_test_split(back, y, train_size=0.7, test_size=0.3, random_state=20)


    # Train Gradient Boosting Classifier
    grb = GradientBoostingClassifier()
    grb.fit(x_train, y_train)
    grb_score = grb.score(x_test, y_test)

    # Feature importances for Gradient Boosting Classifier
    grb_importances = grb.feature_importances_
    grb_importances_df = pd.DataFrame(grb_importances, index=x_train.columns, columns=['Importance'])
    grb_importances_df = grb_importances_df.sort_values('Importance', ascending=False)

    # Permutation importance for Gradient Boosting Classifier
    grb_permutation_importance = permutation_importance(grb, x_test, y_test, n_repeats=10, random_state=42)
    grb_perm_importances_df = pd.DataFrame(grb_permutation_importance.importances_mean, index=x_train.columns,
                                           columns=['Importance'])
    grb_perm_importances_df = grb_perm_importances_df.sort_values('Importance', ascending=False)

    # Train DecisionTreeClassifier
    dtree = DecisionTreeClassifier(max_depth=3, criterion="entropy")
    dtree.fit(x_train, y_train)
    dtree_score = dtree.score(x_test, y_test)

    # Feature importances for DecisionTreeClassifier
    dtree_importances = dtree.feature_importances_
    dtree_importances_df = pd.DataFrame(dtree_importances, index=x_train.columns, columns=['Importance'])
    dtree_importances_df = dtree_importances_df.sort_values('Importance', ascending=False)

    # Permutation importance for DecisionTreeClassifier
    dtree_permutation_importance = permutation_importance(dtree, x_test, y_test, n_repeats=10, random_state=42)
    dtree_perm_importances_df = pd.DataFrame(dtree_permutation_importance.importances_mean, index=x_train.columns,
                                             columns=['Importance'])
    dtree_perm_importances_df = dtree_perm_importances_df.sort_values('Importance', ascending=False)

    # Train ADABoost
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier()
    ada.fit(x_train, y_train)
    ada_score = ada.score(x_test, y_test)

    # Feature importances for ADABoost
    ada_importances = ada.feature_importances_
    ada_importances_df = pd.DataFrame(ada_importances, index=x_train.columns, columns=['Importance'])
    ada_importances_df = ada_importances_df.sort_values('Importance', ascending=False)

    # Permutation importance for ADABoost
    ada_permutation_importance = permutation_importance(ada, x_test, y_test, n_repeats=10, random_state=42)
    ada_perm_importances_df = pd.DataFrame(ada_permutation_importance.importances_mean, index=x_train.columns,
                                           columns=['Importance'])
    ada_perm_importances_df = ada_perm_importances_df.sort_values('Importance', ascending=False)

    return render_template(
        'models_results.html',
        grb_score=grb_score,
        grb_importances=grb_importances_df.to_html(),
        grb_perm_importances=grb_perm_importances_df.to_html(),
        dtree_score=dtree_score,
        dtree_importances=dtree_importances_df.to_html(),
        dtree_perm_importances=dtree_perm_importances_df.to_html(),
        ada_score=ada_score,
        ada_importances=ada_importances_df.to_html(),
        ada_perm_importances=ada_perm_importances_df.to_html()
    )

@app.route('/visualizations',methods=['GET','POST'])
def visualization():
    return render_template("visualizations.html")


@app.route('/visualize_test', methods=['GET','POST'])
def visualize_test():
    data = pd.read_csv('Scenario-B-merged_5s.csv')
    data.columns=data.columns.str.strip()
    print(data.columns)
    print(data.info())
    print(data.isna().sum())
    data = data.replace([-np.inf, np.inf], np.NAN)
    print(data.info())

    def ip_convert(ip):
        ip = str(ip).split('.')
        if ip[0] < '127' and ip[0] > '0':
            return 'A'
        elif ip[0] > '128' and ip[0] < '192':
            return 'B'
        elif ip[0] > '192' and ip[0] < '223':
            return 'C'
        elif ip[0] > '223' and ip[0] < '240':
            return 'D'
        elif ip[0] > '240':
            return 'E'

    data['Source IP'] = data['Source IP'].apply(ip_convert)
    data['Destination IP'] = data['Destination IP'].apply(ip_convert)

    lab = LabelEncoder()

    for i in data.select_dtypes(include='object').columns.values:
        data[i] = lab.fit_transform(data[i])
    print(data.label.value_counts())

    x = []
    for i in data.columns.values:
        data['z-scores'] = (data[i] - data[i].mean()) / data[i].std()
        outliers = np.abs(data['z-scores'] > 3).sum()
        if outliers > 0:
            x.append(i)

    print(len(data))
    thresh = 3
    for i in x[:5]:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]
    print(len(data))

    print(data.columns)
    univariate1 = request.form.get('univariate1')
    univariate2 = request.form.get('univariate2')
    plot_type = request.form.get('univariate3')

    plot_filename = f"{plot_type}_{univariate1}_{univariate2}.png"
    plot_path = f'static/images/{plot_filename}'

    if plot_type == 'histogram':
        fig, ax = plt.subplots()
        data[univariate1].hist(ax=ax)
        ax.set_title('Histogram of ' + univariate1)
        fig.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'boxplot':
        fig = plt.figure()
        sns.boxplot(data=data, x=univariate1, y=univariate2)
        plt.title(f'Boxplot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'density':
        fig = plt.figure()
        sns.kdeplot(data=data, x=univariate1, y=univariate2, fill=True)
        plt.title(f'Density Plot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'violin':
        fig = plt.figure()
        sns.violinplot(data=data, x=univariate1, y=univariate2)
        plt.title(f'Violin Plot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'bar':
        fig = plt.figure()
        data[univariate1].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {univariate1}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'scatter':
        fig = px.scatter(data, x=univariate1, y=univariate2, title='Scatter Plot of {} vs {}'.format(univariate1, univariate2))
        pio.write_image(fig, plot_path, format='png')

    elif plot_type == 'heatmap':
        corr = data.corr()
        fig = plt.figure()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Heatmap of Correlation Matrix')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'contour':
        x = data[univariate1]
        y = data[univariate2]
        x = x[~x.isna()]
        y = y[~y.isna()]
        X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        Z = np.exp(-(X**2 + Y**2))
        fig = plt.figure()
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.title('Contour Plot')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'hexbin':
        fig, ax = plt.subplots()
        hb = ax.hexbin(data[univariate1], data[univariate2], gridsize=50, cmap='inferno')
        plt.colorbar(hb, ax=ax)
        ax.set_title('Hexbin Plot of {} vs {}'.format(univariate1, univariate2))
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'stacked-bar':
        fig = plt.figure()
        data.groupby(univariate1)[univariate2].sum().plot(kind='bar', stacked=True)
        plt.title('Stacked Bar Chart')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'qq-plot':
        fig = plt.figure()
        sm.qqplot(data[univariate1], line='45')
        plt.title(f'Q-Q Plot of {univariate1}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'cumulative':
        fig = plt.figure()
        data[univariate1].cumsum().plot()
        plt.title(f'Cumulative Plot of {univariate1}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'violin-box':
        fig = plt.figure()
        sns.violinplot(data=data, x=univariate1, y=univariate2)
        sns.boxplot(data=data, x=univariate1, y=univariate2, whis=np.inf)
        plt.title(f'Violin with Boxplot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)
    else:
        return "Invalid plot type", 400

    plot_url = f'/static/images/{plot_filename}'
    return render_template("visualizations.html",plot_url=plot_url)

def summary_stats(column1, column2, stat_type):
    data = pd.read_csv('Scenario-B-merged_5s.csv')
    data.columns = data.columns.str.strip()
    print(data.columns)
    print(data.info())
    print(data.isna().sum())
    data = data.replace([-np.inf, np.inf], np.NAN)
    print(data.info())

    def ip_convert(ip):
        ip = str(ip).split('.')
        if ip[0] < '127' and ip[0] > '0':
            return 'A'
        elif ip[0] > '128' and ip[0] < '192':
            return 'B'
        elif ip[0] > '192' and ip[0] < '223':
            return 'C'
        elif ip[0] > '223' and ip[0] < '240':
            return 'D'
        elif ip[0] > '240':
            return 'E'

    data['Source IP'] = data['Source IP'].apply(ip_convert)
    data['Destination IP'] = data['Destination IP'].apply(ip_convert)

    lab = LabelEncoder()

    for i in data.select_dtypes(include='object').columns.values:
        data[i] = lab.fit_transform(data[i])
    print(data.label.value_counts())

    x = []
    for i in data.columns.values:
        data['z-scores'] = (data[i] - data[i].mean()) / data[i].std()
        outliers = np.abs(data['z-scores'] > 3).sum()
        if outliers > 0:
            x.append(i)

    print(len(data))
    thresh = 3
    for i in x[:5]:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]
    print(len(data))

    print(data.columns)

    result = ""
    if stat_type == "mean":
        result = f"Mean of {column1}: {data[column1].mean()}"
    elif stat_type == "median":
        result = f"Median of {column1}: {data[column1].median()}"
    elif stat_type == "mode":
        result = f"Mode of {column1}: {data[column1].mode()[0]}"
    elif stat_type == "variance":
        result = f"Variance of {column1}: {data[column1].var()}"
    elif stat_type == "std_dev":
        result = f"Standard Deviation of {column1}: {data[column1].std()}"
    elif stat_type == "kurtosis":
        result = f"Kurtosis of {column1}: {stats.kurtosis(data[column1].dropna())}"
    elif stat_type == "skewness":
        result = f"Skewness of {column1}: {stats.skew(data[column1].dropna())}"
    elif stat_type == "range":
        result = f"Range of {column1}: {data[column1].max() - data[column1].min()}"
    elif stat_type == "iqr":
        result = f"Interquartile Range of {column1}: {stats.iqr(data[column1].dropna())}"
    elif stat_type == "t-test":
        t_stat, p_val = stats.ttest_ind(data[column1].dropna(), data[column2].dropna())
        result = f"T-Test: t-statistic={t_stat}, p-value={p_val}"
    elif stat_type == "anova":
        groups = [data[column1][data[column2] == g] for g in data[column2].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        result = f"ANOVA: F-statistic={f_stat}, p-value={p_val}"
    elif stat_type == "kruskal-wallis":
        groups = [data[column1][data[column2] == g] for g in data[column2].unique()]
        h_stat, p_val = stats.kruskal(*groups)
        result = f"Kruskal-Wallis Test: H-statistic={h_stat}, p-value={p_val}"
    elif stat_type == "pearson-correlation":
        corr, _ = stats.pearsonr(data[column1].dropna(), data[column2].dropna())
        result = f"Pearson Correlation: {corr}"
    elif stat_type == "spearman-correlation":
        corr, _ = stats.spearmanr(data[column1].dropna(), data[column2].dropna())
        result = f"Spearman Correlation: {corr}"
    elif stat_type == "lin-regression":
        from sklearn.linear_model import LinearRegression
        X = data[[column1]]
        y = data[column2]
        model = LinearRegression().fit(X, y)
        result = f"Linear Regression: Coefficient={model.coef_[0]}, Intercept={model.intercept_}"
    return result

@app.route('/stat_test', methods=['GET','POST'])
def stat_test():
    stats1 = request.form.get('stats1')
    stats2 = request.form.get('stats2')
    stats3 = request.form.get('stats3')
    if stats3:
        summary = summary_stats(stats1, stats2, stats3)
    else:
        summary = "Please select a summary statistic or test."
    return render_template('visualizations.html', summary=summary)


if __name__=='__main__':
    app.run()