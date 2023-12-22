import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import pandas as pd
import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
import plotly.subplots
import seaborn as sns
import plotly.graph_objects as go


import math
import os
from plotly.subplots import make_subplots


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime



def separate_dataframe(df, date_column, reference_date):
    # Convert the date column to datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Separate the dataframe into two based on the reference date
    before_df = df[df[date_column] < reference_date]
    after_df = df[df[date_column] >= reference_date]
    
    return before_df, after_df




def perform_anova(list1, list2):
    # Perform the ANOVA test
    f_statistic, p_value = f_oneway(list1, list2)
    
    # Return the significance (p-value)
    return f_statistic, p_value


def calculate_z_score(list1, list2):
    # Convert the lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Calculate the t-statistic and p-value using the ttest_ind function from scipy.stats
    t_statistic, p_value = stats.ttest_ind(array1, array2)
    
    # Calculate the z-score using the t-statistic and sample sizes
    n1 = len(array1)
    n2 = len(array2)
    z_score = (t_statistic * np.sqrt(n1 + n2 - 2)) / np.sqrt(n1 + n2)
    
    return z_score, p_value




def merging_mobile_values_computer(df,language):
    language_subset = [x for x in df.columns if x.startswith(language)]
    language_dictionary = {}
    if len(language_subset) == 2:
        print(df[language_subset[0]]['topics']['STEM.Earth and environment']['sum'].values)
        list_1 = list(df[language_subset[0]]['topics']['STEM.Earth and environment']['sum'].values())
        list_2 = list(df[language_subset[1]]['topics']['STEM.Earth and environment']['sum'].values())
        n1 = len(list(df[language_subset[0]]['topics']['STEM.Earth and environment']['sum']))
        n2 = len(list(df[language_subset[1]]['topics']['STEM.Earth and environment']['sum']))
        divided_list_1 = [element / n1 for element in list_1]
        divided_list_2 = [element / n2 for element in list_2]
        values = [a+b for a,b in zip(divided_list_1,divided_list_2)]
        print(len(values))
        keys = list(df[language_subset[0]]['topics']['STEM.Earth and environment']['sum'].keys())
        print(len(keys))
        language_dictionary[language] = values
        language_dictionary['date'] = keys
        language_df = pd.DataFrame(language_dictionary)

        return language_df
    elif len(language_subset) == 1:
        values = df[language_subset[0]]['topics']['STEM.Earth and environment']['sum'].values()
        print(len(values))
        keys = list(df[language_subset[0]]['topics']['STEM.Earth and environment']['sum'].keys())
        print(len(keys))
        language_dictionary[language] = values
        language_dictionary['date'] = keys
        language_df = pd.DataFrame(language_dictionary)
        return language_df
    

# Define a function to extract data into a DataFrame for each language
def extract_language_data(language_key, data):
    # Extract pageviews for the language
    pageviews_data = data[language_key]['sum']
    
    # Prepare the data for DataFrame creation
    rows_list = [{'date': date, 'pageviews': pageviews} for date, pageviews in pageviews_data.items()]
    
    # Create and return the DataFrame
    return pd.DataFrame(rows_list)

def standardization(df, date_column):

    for col in df.columns:
        if col != date_column:
            col_mean = df[col].mean()
            col_std = df[col].std()
            df.loc[:, col] = (df[col] - col_mean) / col_std

    return df

def plot_distribution_with_means(pre_values, post_values, pre_label='Pre-COVID', post_label='Post-COVID'):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(pre_values, kde=True, stat="density", linewidth=0, color="red", label=pre_label)
    sns.histplot(post_values, kde=True, stat="density", linewidth=0, color="blue", label=post_label)
    
    # plot the means
    plt.axvline(x=np.mean(pre_values), color="red", linestyle="--", label=f'{pre_label} Mean')
    plt.axvline(x=np.mean(post_values), color="blue", linestyle="--", label=f'{post_label} Mean')
    
    
    plt.title('Distribution of Total Pageviews Pre and Post COVID-19', fontsize=16)
    plt.xlabel('Total Pageviews', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    
    plt.show()

def plot_distribution_with_means_plotly(pre_covid_values, post_covid_values, title, write):
    fig = go.Figure()

    # Add histogram for pre-COVID values
    fig.add_trace(go.Histogram(
        x=pre_covid_values,
        name='Pre-COVID',
        marker_color='red',
        opacity=0.5,
        histnorm='probability density'
    ))

    # Add histogram for post-COVID values
    fig.add_trace(go.Histogram(
        x=post_covid_values,
        name='Post-COVID',
        marker_color='blue',
        opacity=0.5,
        histnorm='probability density'
    ))

    # Calculate and add a line for the mean of the pre-COVID values
    fig.add_trace(go.Scatter(
        x=[np.mean(pre_covid_values)] * 2,
        y=[0, max(np.histogram(pre_covid_values, bins=20, density=True)[0])],  # Scale the line to the histogram height
        mode='lines',
        name='Pre-COVID Mean',
        line=dict(color='red', width=3, dash='dash')
    ))

    # Calculate and add a line for the mean of the post-COVID values
    fig.add_trace(go.Scatter(
        x=[np.mean(post_covid_values)] * 2,
        y=[0, max(np.histogram(post_covid_values, bins=20, density=True)[0])],  # Scale the line to the histogram height
        mode='lines',
        name='Post-COVID Mean',
        line=dict(color='blue', width=3, dash='dash')
    ))

    
    fig.update_layout(
        title_text='Distributions of Pre and Post COVID Values for {}'.format(title),
        xaxis_title='Value',
        yaxis_title='Density',
        barmode='overlay'  
    )
    
    
    fig.update_layout(
        autosize=False,
        width=900,
        height=600,
        showlegend=True)
    if write == True:
        pio.write_html(fig, "Total Pageviews in StemEarth.html")
    
    fig.show()


def plot_language_distributions(pre_covid, post_covid, languages):
    num_languages = len(languages)
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))

    # Flatten the axis array for easy iteration
    axs = axs.flatten()
    
    # We will use the last subplot for the legend
    legend_ax = axs[-1]  # The last subplot

    for i, language in enumerate(languages):
        ax = axs[i]

        # Plotting for each language
        sns.histplot(pre_covid[language], kde=True, stat="density", linewidth=0, color="red", ax=ax)
        sns.histplot(post_covid[language], kde=True, stat="density", linewidth=0, color="blue", ax=ax)

        # Plot the means
        pre_mean = np.mean(pre_covid[language])
        post_mean = np.mean(post_covid[language])
        
        pre_line = ax.axvline(x=pre_mean, color="red", linestyle="--")
        post_line = ax.axvline(x=post_mean, color="blue", linestyle="--")

        ax.set_title(language)

    # Remove the unused subplots
    for j in range(num_languages, len(axs)-1):
        fig.delaxes(axs[j])

    # Add overall legends and means legends in the last subplot
    legend_ax.legend([pre_line, post_line ], ['Pre-COVID Mean', 'Post-COVID Mean'], loc='center')
    legend_ax.text(0.5, 0.7, 'Pre-COVID', color='red', ha='center')
    legend_ax.text(0.5, 0.2, 'Post-COVID', color='blue', ha='center')
    legend_ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_language_distributions_plotly(pre_covid, post_covid, languages, write):
    num_languages = len(languages)
    cols = 3  
    rows = -(-num_languages // cols)  

    # Create subplot grid
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=languages)

    # Add histograms and mean lines for each language
    for i, language in enumerate(languages):
        row, col = (i // cols) + 1, (i % cols) + 1

        # Normalized histograms for pre-COVID
        fig.add_trace(go.Histogram(x=pre_covid[language], marker_color='red', opacity=0.6, histnorm='probability density'), row=row, col=col)
        # Normalized histograms for post-COVID
        fig.add_trace(go.Histogram(x=post_covid[language], marker_color='blue', opacity=0.6, histnorm='probability density'), row=row, col=col)
        # Calculate and add mean lines
        fig.add_trace(go.Scatter(x=[np.mean(pre_covid[language])]*2, y=[0, 1], mode='lines', line=dict(color='red', dash='dash')), row=row, col=col)
        fig.add_trace(go.Scatter(x=[np.mean(post_covid[language])]*2, y=[0, 1], mode='lines', line=dict(color='blue', dash='dash')), row=row, col=col)

    fig.update_layout(
        height=900, width=1000,
        title_text="Language Distributions Pre and Post COVID",
        barmode='overlay'
    )

    fig.update_yaxes(title_text="Density")

    # Define the legend once
    fig.data[0].name = 'Pre-COVID'
    fig.data[1].name = 'Post-COVID'
    fig.data[2].name = 'Pre-COVID Mean'
    fig.data[3].name = 'Post-COVID Mean'
    for i in range(4, len(fig.data)):
        fig.data[i].showlegend = False

    # Adjust the legend's position
    fig.update_layout(legend=dict(x=1, y=0, traceorder='normal', orientation='v', xanchor='right', yanchor='bottom'))
    # hide the last subplot
    fig.update_xaxes(visible=False, row=rows, col=cols)
    fig.update_yaxes(visible=False, row=rows, col=cols)
    if write == True:
        pio.write_html(fig, "Language Distributions.html")
    fig.show()


def perform_z_tests(pre_covid, post_covid, languages):
    z_scores = []
    p_values = []
    
    for lang in languages:
        # Perform z-test
        z_score, p_value = calculate_z_score(post_covid[lang], pre_covid[lang])
        z_scores.append(z_score)
        p_values.append(p_value)
    
    return z_scores, p_values

def interactive_z_p_scores(df , title, write):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Z-Scores for {}'.format(title), '-10 Log (P-Values) for {}'.format(title)))

    colors = px.colors.qualitative.Plotly

    # Create a bar plot for Z-Scores vs Languages

    fig.add_trace(go.Bar(x=df['Language'], y=df['Z-Score'], name = 'Z-Score'# color='Language', #color_discrete_sequence=colors,
#                             title='Z-Scores for {}'.format(title)
                            ),row=1, col=1
                 )
    
    fig.add_trace(go.Bar(x=df['Language'], y=df['log(P-Value)'], name = 'P-Value'# color='Language', #color_discrete_sequence=colors,
#                             title='Z-Scores for {}'.format(title)
                            ),row=2, col=1
                 )
    
    fig.add_hline(y=-10*np.log10(0.05), #line_dash="dashdot", line_color="red",
              line=dict(dash='dashdot', color = 'red', width = 3),row=2, col=1,
                  
             )


    fig.update_layout(
        autosize=False,
        width=800,
         height=800)
    fig.update_yaxes(type="log", row=2, col=1)
    if write == True:
        pio.write_html(fig, "Z and P for Each Language.html")

    fig.show()


def plot_pre_post_covid_distributions(dfs, titles, reference_date):
    n_rows = len(dfs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(5, 3 * n_rows), squeeze=False)
    
    for i, (df, title) in enumerate(zip(dfs, titles)):
        # Separate each dataframe into pre and post COVID dataframes
        pre_covid, post_covid = separate_dataframe(df, 'date', reference_date)

        # Extract the values for plotting
        pre_values = pre_covid['total'].values
        post_values = post_covid['total'].values

        ax = axes[i, 0]
        sns.histplot(pre_values, kde=True, stat="density", linewidth=0, color="red", label='Pre-COVID', ax=ax)
        sns.histplot(post_values, kde=True, stat="density", linewidth=0, color="blue", label='Post-COVID', ax=ax)

        # Plot the means
        ax.axvline(x=np.mean(pre_values), color="red", linestyle="--", label='Pre-COVID Mean')
        ax.axvline(x=np.mean(post_values), color="blue", linestyle="--", label='Post-COVID Mean')

        # Set the titles and labels
        ax.set_title(f'Distribution of Total Pageviews for {title} Pre and Post COVID-19', fontsize=8)
        ax.set_xlabel('Total Pageviews', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend()


    plt.tight_layout()
    plt.show()

def plot_distribution_with_means_plotly_traces(pre_covid_values, post_covid_values, title):
#     fig = go.Figure()
    traces = []
    # Add histogram for pre-COVID values
    traces.append(go.Histogram(
        x=pre_covid_values,
        name='Pre-COVID',
        marker_color='red',
        opacity=0.5,
        histnorm='probability density'
    ))

    # Add histogram for post-COVID values
    traces.append(go.Histogram(
        x=post_covid_values,
        name='Post-COVID',
        marker_color='blue',
        opacity=0.5,
        histnorm='probability density'
    ))

    # Calculate and add a line for the mean of the pre-COVID values
    traces.append(go.Scatter(
        x=[np.mean(pre_covid_values)] * 2,
        y=[0, max(np.histogram(pre_covid_values, bins=20, density=True)[0])],  # Scale the line to the histogram height
        mode='lines',
        name='Pre-COVID Mean',
        line=dict(color='red', width=3, dash='dash')
    ))

    # Calculate and add a line for the mean of the post-COVID values
    traces.append(go.Scatter(
        x=[np.mean(post_covid_values)] * 2,
        y=[0, max(np.histogram(post_covid_values, bins=20, density=True)[0])],  # Scale the line to the histogram height
        mode='lines',
        name='Post-COVID Mean',
        line=dict(color='blue', width=3, dash='dash')
    ))

    return traces


def create_layout_button(k, customer, Lc, n):

    """
    Create a button for the interactive plot
    input: k = index of the button
            customer = button options
            Lc = number options
    return: dictionary with the button options
    """


    visibility= [False]*n*Lc
    for tr in np.arange(n*k, n*k+n):
        visibility[tr] =True
    return dict(label = customer, method = 'update', args = [{'visible': visibility,'title': customer,'showlegend': True}])  


def plot_pre_post_covid_distributions_plotly(dfs, titles, reference_date, write):
    # Define the number of rows needed for subplots based on the number of dataframes
    n_rows = len(dfs)
    fig = make_subplots()#make_subplots(rows=n_rows, cols=1, subplot_titles=titles)
    all_traces = []
    for i, df in enumerate(dfs, start=1):
        # Separate each dataframe into pre and post COVID dataframes
        pre_covid, post_covid = separate_dataframe(df, 'date', reference_date)
        pre_covid_values = pre_covid['total'].values
        post_covid_values = post_covid['total'].values

        # Extract the values for plotting
        all_traces.extend(plot_distribution_with_means_plotly_traces(pre_covid_values, post_covid_values, titles[i-1]))
#     fig = make_subplots()
    for i in range(0,len(all_traces)):
        fig.add_trace(all_traces[i])

    Ld=len(fig.data)
    Lc =len(titles)
    for k in range(4, Ld):
        fig.update_traces(visible=False, selector = k)

    fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active = 0,
        buttons = [create_layout_button(k, customer, Lc, 4) for k, customer in enumerate(titles)],
        x = 0.58,
        xanchor = 'left',
        y = 1.23,
        yanchor = 'top',
        )
    ])
    
    fig.update_layout(
    title_text='Distributions of Pre and Post COVID Values for ',#.format(title),
    xaxis_title='Value',
    yaxis_title='Density',
    barmode='overlay'  
    )
    
    
    
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        showlegend=True)

    if write == True:
        pio.write_html(fig, "Distributions for different terms.html")

    fig.show()
        

def perform_z_tests_and_plot(dfs, titles, reference_date):
    z_scores = []
    p_values = []
    results = []

    # Perform z-test for each dataframe and collect results
    for df, title in zip(dfs, titles):
        # Separate the dataframe into pre and post COVID dataframes
        pre_covid, post_covid = separate_dataframe(df, 'date', reference_date)

        # Extract the total values
        pre_values = pre_covid['total'].values
        post_values = post_covid['total'].values

        # Calculate the z-score and p-value using a t-test (as a stand-in for the z-test)
        z_score, p_value = calculate_z_score(pre_values, post_values)

        # Append the results to the lists
        z_scores.append(z_score)
        p_values.append(p_value)
        results.append((title, z_score, p_value))

    # Define a color palette
    palette = sns.color_palette('hsv', len(titles))

    # Plotting the z-scores and p-values
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Scatter plot for Z-Scores
    sns.scatterplot(x=[r[0] for r in results], y=[r[1] for r in results], ax=ax[0], hue=titles, palette=palette, s=100)
    ax[0].set_title('Z-Scores for Each DataFrame')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[0].axhline(0, color='grey', lw=1, linestyle='--')
    ax[0].legend(loc='upper left')
    ax[0].get_legend().remove()

    # Scatter plot for P-Values (log scale)
    negative_log_p_values = [-np.log10(p) if p > 0 else 50 for p in p_values]  # Cap the values for visibility
    sns.scatterplot(x=[r[0] for r in results], y=negative_log_p_values, ax=ax[1], hue=titles, palette=palette, s=100)
    ax[1].set_title('-log10(P-Values) for Each DataFrame')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[1].axhline(-np.log10(0.05), color='red', lw=1, linestyle='--', label='Significance Level (p=0.05)')
    ax[1].get_legend().remove()

    plt.tight_layout()
    plt.show()

    return results



def find_common_columns(dataframes):

    if not dataframes:
        return set()

    # Get the columns of the first dataframe as a starting point
    common_columns = set(dataframes[0].columns)

    # Intersect with the columns of the remaining dataframes
    for df in dataframes[1:]:
        common_columns = common_columns.intersection(set(df.columns))

    return common_columns




def perform_z_tests_by_dataframe_and_plot(dfs, titles, reference_date, write):
    # Initialize a dictionary to hold results for each dataframe
    dataframe_results = {}
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Z-Scores for Article Views',#.format(title),
                                                        '-10 Log (P-Values) for Article Views'#.format(title)
                                                       )
                       )
    # Iterate over each dataframe and its title
    for df, title in zip(dfs, titles):
        # Separate the dataframe into pre and post COVID dataframes
        pre_covid, post_covid = separate_dataframe(df.drop(columns=['total']), 'date', reference_date)
        df_results = []

        # Iterate over each column (language) in the dataframe
        for column in pre_covid.columns:
            if column != 'date':
                # Extract the values for each language
                pre_values = pre_covid[column].values
                post_values = post_covid[column].values

                # Calculate the z-score and p-value for each language
                z_score, p_value = calculate_z_score(pre_values, post_values)
                df_results.append({'Language': column, 'Z-Score': z_score, 'P-Value': p_value})
        
        
        
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(df_results)
        
        
        fig.add_trace(go.Bar(x=results_df['Language'], y=results_df['Z-Score'], name = 'Z-Score'# color='Language', #color_discrete_sequence=colors,
#                             title='Z-Scores for {}'.format(title)
                            ),row=1, col=1
                 )
        
#         # Plotting the z-scores and p-values for each dataframe
        fig.add_hline(y=0, 
                      line=dict(dash='dashdot', color = 'grey', width = 3),row=1, col=1,

                     )


        # Scatter plot for P-Values (log scale)
        results_df['Negative Log P-Value'] = -np.log10(results_df['P-Value'].clip(lower=1e-50))  # Avoid -inf values

        fig.add_trace(go.Bar(x=results_df['Language'], y=results_df['Negative Log P-Value'], name = 'P-Value'# color='Language', #color_discrete_sequence=colors,
                            ),row=2, col=1
                 )
        fig.add_hline(y=-10*np.log10(0.05), 
                  line=dict(dash='dashdot', color = 'red', width = 3),row=2, col=1,
                 )
        dataframe_results[title] = results_df
        
    Ld=len(fig.data)
    Lc =len(titles)
    for k in range(2, Ld):
        fig.update_traces(visible=False, selector = k)

    fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active = 0,
        buttons = [create_layout_button(k, customer, Lc, 2) for k, customer in enumerate(titles)],
        x = 0.7,
        xanchor = 'left',
        y = 1.06,
        yanchor = 'top',
        )
    ])
    
        
        
    fig.update_layout(
    autosize=False,
    width=800,
     height=800)
    
    
    fig.update_yaxes(range=[-50,30], row=1, col=1)
    fig.update_yaxes(range=[-10,60], row=2, col=1)
    
    if write == True:
        pio.write_html(fig, "ZandP for different articles.html")

    fig.show()

    return dataframe_results

def perform_z_tests_by_dataframe_and_plot2(dfs, titles, reference_date):
    # Initialize a dictionary to hold results for each dataframe
    dataframe_results = {}

    # Determine unique languages across all dataframes
    unique_languages = set()
    for df in dfs:
        unique_languages.update(df.drop(columns=['total', 'date']).columns)
    
    # Assign a unique color to each language
    language_colors = {lang: color for lang, color in zip(unique_languages, sns.color_palette('hsv', len(unique_languages)))}

    # Iterate over each dataframe and its title
    for df, title in zip(dfs, titles):
        # Separate the dataframe into pre and post COVID dataframes
        pre_covid, post_covid = separate_dataframe(df.drop(columns=['total']), 'date', reference_date)
        df_results = []

        # Iterate over each column (language) in the dataframe
        for column in pre_covid.columns:
            if column != 'date':
                # Extract the values for each language
                pre_values = pre_covid[column].values
                post_values = post_covid[column].values

                # Calculate the z-score and p-value for each language
                z_score, p_value = calculate_z_score(pre_values, post_values)
                df_results.append((column, z_score, p_value, language_colors[column]))

        dataframe_results[title] = df_results

    # Plotting the z-scores and p-values for each dataframe
    n_dfs = len(dataframe_results)
    fig, axes = plt.subplots(n_dfs, 2, figsize=(14, 7 * n_dfs))

    for i, (title, results) in enumerate(dataframe_results.items()):
        # Unpack the results for plotting
        languages, z_scores, p_values, colors = zip(*results)
        negative_log_p_values = [-np.log10(p) if p > 0 else 50 for p in p_values]  # Cap values for visibility

        # Scatter plot for Z-Scores
        ax_z = axes[i, 0]
        sns.scatterplot(x=languages, y=z_scores, ax=ax_z, hue=languages, palette=colors, s=100)
        ax_z.set_title(f'Z-Scores for {title}')
        ax_z.axhline(0, color='grey', lw=1, linestyle='--')
        ax_z.get_legend().remove()
        ax_z.set_xticks([])
        

        # Scatter plot for P-Values (log scale)
        ax_p = axes[i, 1]
        sns.scatterplot(x=languages, y=negative_log_p_values, ax=ax_p, hue=languages, palette=colors, s=100)
        ax_p.set_title(f'-log10(P-Values) for {title}')
        ax_p.axhline(-np.log10(0.05), color='red', lw=1, linestyle='--', label='Significance Level (p=0.05)')
        ax_p.get_legend().remove()
        ax_p.set_xticklabels(ax_p.get_xticklabels(), rotation=90 , size=5)
        ax_p.set_xticks([])
        

        ax_p.legend(title='Languages', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    return dataframe_results




def summarize_significant_changes(results):
    significant_increase = []
    significant_decrease = []

    # Threshold for significance
    significance_level = 0.05


    for dataframe, df_results in results.items():
        for language, z_score, p_value, _ in df_results:
            if p_value < significance_level:
                if z_score > 0:
                    # Significant increase
                    significant_increase.append((dataframe, language, z_score, p_value))
                elif z_score < 0:
                    # Significant decrease
                    significant_decrease.append((dataframe, language, z_score, p_value))


    increase_df = pd.DataFrame(significant_increase, columns=['DataFrame', 'Language', 'Z-Score', 'P-Value'])
    decrease_df = pd.DataFrame(significant_decrease, columns=['DataFrame', 'Language', 'Z-Score', 'P-Value'])

    return increase_df, decrease_df


def plot_heatmap_of_changes(increase_df, decrease_df):
    # Pivot dataframes for heatmap plotting
    pivot_increase = increase_df.pivot(index='DataFrame', columns='Language', values='Z-Score')
    pivot_decrease = decrease_df.pivot(index='DataFrame', columns='Language', values='Z-Score')     

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Heatmap for significant increase
    sns.heatmap(pivot_increase, annot=True, cmap='Greens', ax=axes[0])
    axes[0].set_title('Significant Increase Post-COVID')

    # Heatmap for significant decrease
    sns.heatmap(pivot_decrease, annot=True, cmap='Reds', ax=axes[1])
    axes[1].set_title('Significant Decrease Post-COVID')

    plt.tight_layout()
    plt.show()



def plot_heatmap_of_changes_plotly(increase_df, decrease_df, write):
    # Pivot dataframes for heatmap plotting and round Z-scores
    pivot_increase = increase_df.pivot(index='DataFrame', columns='Language', values='Z-Score').round(2)
    pivot_decrease = decrease_df.pivot(index='DataFrame', columns='Language', values='Z-Score').round(2)

    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Significant Increase Post-COVID", "Significant Decrease Post-COVID"),
                        vertical_spacing=0.12)

    # Heatmap for significant increase with rounded Z-scores and corrected colorscale
    fig.add_trace(
        go.Heatmap(
            z=pivot_increase.values, 
            x=pivot_increase.columns, 
            y=pivot_increase.index, 
            colorscale='Greens', 
            reversescale=True, 
            zmin=pivot_increase.values.min(),
            zmax=pivot_increase.values.max(),  
            text=pivot_increase.values,
            texttemplate="%{text}",
            hoverinfo="text",
            colorbar_x=1.01

        ),
        row=1, col=1
    )

    # Heatmap for significant decrease with rounded Z-scores
    fig.add_trace(
        go.Heatmap(
            z=pivot_decrease.values, 
            x=pivot_decrease.columns, 
            y=pivot_decrease.index, 
            colorscale='Reds', 
            zmin=pivot_decrease.values.min(),  # Ensure the scale includes the min value
            zmax=pivot_decrease.values.max(),  # Ensure the scale includes the max value
            text=pivot_decrease.values,
            texttemplate="%{text}",
            hoverinfo="text",
            colorbar_x=1.11
        ),
        row=2, col=1
    )


    fig.update_layout(
        height=700, 
        width=900, 
        title_text="Significant Changes Post-COVID",
        title_font_size=24,
        font=dict(size=12),
        margin=dict(t=100, l=70, b=70, r=80),  
        coloraxis_colorbar=dict(title="Z-Score" )
        
    )
    

    # Update axes titles
    
    fig.update_xaxes(title_text="Language", row=2, col=1)
    fig.update_yaxes(title_text="DataFrame", row=1, col=1)
    fig.update_yaxes(title_text="DataFrame", row=2, col=1)
    if write == True:
        pio.write_html(fig, "Significant Changes Heatmap.html")

    fig.show()


#######################################################################

def preprocess(data, threshold = 0.75):
    
    data = data.drop('Area', axis=1)

    # Extracting the start time from the 'MTU' column and setting it as index
    data['MTU'] = data['MTU'].str.extract(r'(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})')
    data['MTU'] = pd.to_datetime(data['MTU'], format="%d.%m.%Y %H:%M")
    data.set_index('MTU', inplace=True)
    
    columns_to_rewrite = [col for col in data.columns]

    data['Coal'] = data[['Fossil Brown coal/Lignite  - Actual Aggregated [MW]',\
                         'Fossil Coal-derived gas  - Actual Aggregated [MW]',\
                         'Fossil Hard coal  - Actual Aggregated [MW]',\
                         'Fossil Peat  - Actual Aggregated [MW]']].sum(axis=1)
    data['Oil'] = data[[ 'Fossil Oil  - Actual Aggregated [MW]',\
                         'Fossil Oil shale  - Actual Aggregated [MW]']].sum(axis=1)
    data['Gas'] = data[ 'Fossil Gas  - Actual Aggregated [MW]']
    data['Biomass'] = data['Biomass  - Actual Aggregated [MW]']
    data['Geothermal'] = data['Geothermal  - Actual Aggregated [MW]']
    data['Hydro'] = data[['Hydro Pumped Storage  - Actual Aggregated [MW]',\
                          'Hydro Run-of-river and poundage  - Actual Aggregated [MW]',\
                          'Hydro Water Reservoir  - Actual Aggregated [MW]',\
                          'Marine  - Actual Aggregated [MW]']].sum(axis=1)
    data['Nuclear'] = data['Nuclear  - Actual Aggregated [MW]']
    data['Solar'] = data['Solar  - Actual Aggregated [MW]']
    data['Waste'] = data['Waste  - Actual Aggregated [MW]']
    data['Wind'] = data[['Wind Offshore  - Actual Aggregated [MW]',\
                         'Wind Onshore  - Actual Aggregated [MW]']].sum(axis=1)
    data['Renewable'] = data[['Biomass', 'Geothermal', 'Hydro', 'Solar', 'Waste', 'Wind', 'Other renewable  - Actual Aggregated [MW]']].sum(axis=1)
    data['Nonrenewable'] = data[['Coal', 'Gas', 'Oil', 'Nuclear', 'Other  - Actual Aggregated [MW]']].sum(axis=1)
    data['Total'] = data[['Renewable','Nonrenewable']].sum(axis = 1)
    
    data['Renewable_Percent'] = data['Renewable'] / data['Total']
    data.drop(columns=columns_to_rewrite, inplace=True)

    # Handling missing values and non-numeric strings
    for column in data.columns:
        # Replacing non-numeric strings and 'n/e' with NaN
        data[column] = data[column].replace('n/e', np.nan).astype(float)

        # Checking the proportion of non-missing values
        non_missing_ratio = data[column].notna().mean()
        if non_missing_ratio > threshold:
            # If more than 75% values are non-missing, fill missing values
            # Using forward fill method for numerical data
            data[column] = data[column].fillna(method='ffill')
        else:
            # If less than 75% values are present, set missing values to 0
            data[column] = data[column].fillna(0)

    return data


def get_useful_interventions(interventions):
    useful_interventions = interventions.loc[[0,2,4,6,9,10]]
    useful_interventions['Country'] = ['France', 'Germany', 'Netherlands', 'Serbia', 'Spain', 'Finland']
    for c in useful_interventions.columns[1:8]:
        useful_interventions[c] = pd.to_datetime(useful_interventions[c])
    useful_interventions.set_index('Country', inplace = True)
    return useful_interventions

def analysis(energy_data, country, interventions):
    energy_data.index = pd.to_datetime(energy_data.index)
    traces = []
    is_second_y = []
#     t_stat_totals = []
    p_value_totals =[]
    p_value_renewables = []
    events_exist = []
    for event in interventions.columns[1:]:
        if country in interventions.index:
            event_date = interventions.loc[country, event]
            if pd.notna(event_date):
                pre_event_data = energy_data[energy_data.index < event_date].iloc[-7:]
                post_event_data = energy_data[energy_data.index >= event_date].iloc[0:7]
                
                ###
                t_stat_total, p_value_total = stats.ttest_ind(pre_event_data['Total'], post_event_data['Total'])
                t_stat_renewable, p_value_renewable = stats.ttest_ind(pre_event_data['Renewable'], post_event_data['Renewable'])
                p_value_renewables.append(p_value_renewable)
                p_value_totals.append(p_value_total)
                events_exist.append(event)
                ###
                
                for energy_type in ['Total']:
                    mean_pre = pre_event_data[energy_type].mean()*24/1000
                    mean_post = post_event_data[energy_type].mean()*24/1000
                    sem_pre = pre_event_data[energy_type].sem()*24/1000
                    sem_post = post_event_data[energy_type].sem()*24/1000

                    traces.append(go.Scatter(
                        x=[f'Pre-{event}', f'Post-{event}'],
                        y=[mean_pre, mean_post],
                        error_y=dict(type='data', array=[sem_pre, sem_post], visible=True),
                        name=f'{country} {energy_type} during {event}',
                        showlegend=False,
                        marker=dict(
                            size=8,
                            color='orangered'),
                        line=dict(color='orangered', width=2.5)
                    ))
                    is_second_y.append(False)
                    
                for energy_type in ['Renewable']:
                    mean_pre = pre_event_data[energy_type].mean()*24/1000
                    mean_post = post_event_data[energy_type].mean()*24/1000
                    sem_pre = pre_event_data[energy_type].sem()*24/1000
                    sem_post = post_event_data[energy_type].sem()*24/1000

                    traces.append(go.Scatter(
                        x=[f'Pre-{event}', f'Post-{event}'],
                        y=[mean_pre, mean_post],
                        error_y=dict(type='data', array=[sem_pre, sem_post], visible=True),
                        name=f'{country} {energy_type} during {event}',
                        showlegend=False,
                        marker=dict(
                            
                            size=8,
                            color='dodgerblue', # one of plotly colorscales
                            #showscale=True
                            ),
                        line = dict(color='dodgerblue', width=2.5, dash='dash')
                    ))
                    is_second_y.append(True)

    return traces, is_second_y, dict(zip(events_exist, p_value_totals)), dict(zip(events_exist, p_value_renewables))



def resample_to_d(country):
    resampled_mean = country[country.columns[:-1]].resample('h').mean().resample('d').sum()/1000
    country[country.columns[:-1]] = resampled_mean
    country.dropna(inplace=True)
    country['Month']=country.index.month
    country['DayofMonth']=country.index.day
    country['DayofWeek']=country.index.dayofweek
    country['Year']=country.index.year


def add_m_y_choose(fig):
    fig.update_layout(

        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),)
    

def create_layout_button_2(k, customer, Lc, n):

    """
    Create a button for the interactive plot
    input: k = index of the button
            customer = button options
            Lc = number options
    return: dictionary with the button options
    """

    visibility= [False]*n*Lc
    for tr in np.arange(n*k, n*k+n):
        visibility[tr] =True
    return dict(label = customer, method = 'update', args = [{'visible': visibility,'title': customer,'showlegend': False}])  

def create_layout_button_3(k, customer, Lc, n):

    visibility= [False]*n*Lc
    for tr in np.arange(n*k, n*k+n):
        visibility[tr] =True
    return dict(label = customer, method = 'update', args = [{'visible': visibility,'title': False,'showlegend': True}])  


def cal_average_percent(df,time_begin, time_end):
    last_year_begin = datetime(time_begin.year - 1, time_begin.month, time_begin.day)
    next_year_begin = datetime(time_begin.year + 1, time_begin.month, time_begin.day)
    last_year_end = datetime(time_end.year - 1, time_end.month, time_end.day)
    next_year_end = datetime(time_end.year + 1, time_end.month, time_end.day)
    
    
    df1 = df.loc[last_year_begin:last_year_end]
    df2 = df.loc[time_begin:time_end]
    df3 = df.loc[next_year_begin:next_year_end]
    percent_by_year = pd.concat([df1.mean(),df2.mean(),df3.mean()], axis =1)
    percent_by_year.columns = ['2019','2020', '2021']
    return percent_by_year


def get_percent(df):
    new_df = df[['Renewable_Percent']].copy()
    for resource in [ 'Coal', 'Oil', 'Gas', 'Biomass', 'Geothermal', 'Hydro', 'Nuclear', 'Solar', 'Waste', 'Wind']:
        new_df.loc[:,resource] = (df[resource]/df['Total'])
#         new_df.loc[:,resource] = (df[resource])

    return new_df


def drop_rows_with_zero(df):

    return df[~(df == 0).any(axis=1)]

def cal_anova_results(df,time_begin, time_end):
    last_year_begin = datetime(time_begin.year - 1, time_begin.month, time_begin.day)
    next_year_begin = datetime(time_begin.year + 1, time_begin.month, time_begin.day)
    last_year_end = datetime(time_end.year - 1, time_end.month, time_end.day)
    next_year_end = datetime(time_end.year + 1, time_end.month, time_end.day)
    
    
    anova_results = {}
    df1 = df.loc[last_year_begin:last_year_end]
    df2 = df.loc[time_begin:time_end]
    df3 = df.loc[next_year_begin:next_year_end]
    # Perform ANOVA for each energy source
    for source in df1.columns:
        f_stat, p_value = f_oneway(df1[source], 
                                   df2[source], 
                                   df3[source]
                                  )
        anova_results[source] = {"F-Statistic": f_stat, "p-value": p_value}

    # Convert the results dictionary to a DataFrame for better visualization
    return pd.DataFrame.from_dict(anova_results, orient='index')


def process_OxCGRT(OxCGRT_data):
    OxCGRT_data_countries = OxCGRT_data[(OxCGRT_data['CountryName'] == 'France' ) | (OxCGRT_data['CountryName'] ==  'Finland') | (OxCGRT_data['CountryName'] == 'Germany' ) |(OxCGRT_data['CountryName'] == 'Netherlands' ) |
                      (OxCGRT_data['CountryName'] == 'Serbia' ) |(OxCGRT_data['CountryName'] == 'Spain' )].copy()
    OxCGRT_data_countries.drop(columns= ['E1_Income support', 'E1_Flag',
           'E2_Debt/contract relief', 'E3_Fiscal measures',
           'E4_International support', 'H1_Public information campaigns',
           'H1_Flag', 'H2_Testing policy', 'H3_Contact tracing',
           'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
           'H6M_Facial Coverings', 'H6M_Flag', 'H7_Vaccination policy', 'H7_Flag',
           'H8M_Protection of elderly people', 'H8M_Flag',
           'V1_Vaccine Prioritisation (summary)',
           'V2A_Vaccine Availability (summary)',
           'V2B_Vaccine age eligibility/availability age floor (general population summary)',
           'V2C_Vaccine age eligibility/availability age floor (at risk summary)',
           'V2D_Medically/ clinically vulnerable (Non-elderly)', 'V2E_Education',
           'V2F_Frontline workers  (non healthcare)',
           'V2G_Frontline workers  (healthcare)',
           'V3_Vaccine Financial Support (summary)',
           'V4_Mandatory Vaccination (summary)', 'ConfirmedCases', 'CountryCode', 'RegionName', 'RegionCode',
           'Jurisdiction', 'ConfirmedCases',
           'ConfirmedDeaths', 'MajorityVaccinated', 'PopulationVaccinated', 'EconomicSupportIndex', 'GovernmentResponseIndex_Average'], inplace= True)
    OxCGRT_data_countries['Date'] = pd.to_datetime(OxCGRT_data_countries['Date'], format='%Y%m%d')
    OxCGRT_data_countries.loc[: , 'C1M'] = (100* (OxCGRT_data_countries['C1M_School closing'] + (OxCGRT_data_countries['C1M_Flag'] == 1) * 0.5))/ (OxCGRT_data_countries['C1M_School closing'].max() + ((OxCGRT_data_countries['C1M_Flag'] == 1) * 0.5) )
    OxCGRT_data_countries.loc[: , 'C2M'] = (100*(OxCGRT_data_countries['C2M_Workplace closing'] + (OxCGRT_data_countries['C2M_Flag'] == 1) * 0.5)) / (OxCGRT_data_countries['C2M_Workplace closing'].max() + ((OxCGRT_data_countries['C2M_Flag'] == 1) * 0.5))
    OxCGRT_data_countries.loc[: , 'C3M'] = (100*(OxCGRT_data_countries['C3M_Cancel public events'] + (OxCGRT_data_countries['C3M_Flag'] == 1) * 0.5)) / (OxCGRT_data_countries['C3M_Cancel public events'].max() + ((OxCGRT_data_countries['C3M_Flag'] == 1) * 0.5))
    OxCGRT_data_countries.loc[: , 'C4M'] = (100*(OxCGRT_data_countries['C4M_Restrictions on gatherings'] + (OxCGRT_data_countries['C4M_Flag'] == 1) * 0.5) )/ (OxCGRT_data_countries['C4M_Restrictions on gatherings'].max() + ((OxCGRT_data_countries['C4M_Flag'] == 1) * 0.5))
    OxCGRT_data_countries.loc[: , 'C5M'] = (100*(OxCGRT_data_countries['C5M_Close public transport'] + (OxCGRT_data_countries['C5M_Flag'] == 1) * 0.5)) / (OxCGRT_data_countries['C5M_Close public transport'].max() + ((OxCGRT_data_countries['C5M_Flag'] == 1) * 0.5))
    OxCGRT_data_countries.loc[: , 'C6M'] = (100*(OxCGRT_data_countries['C6M_Stay at home requirements'] + (OxCGRT_data_countries['C6M_Flag'] == 1) * 0.5) )/ (OxCGRT_data_countries['C6M_Stay at home requirements'].max() + ((OxCGRT_data_countries['C6M_Flag'] == 1) * 0.5))
    OxCGRT_data_countries.loc[: , 'C7M'] = (100*(OxCGRT_data_countries['C7M_Restrictions on internal movement'] + (OxCGRT_data_countries['C7M_Flag'] == 1) * 0.5)) / (OxCGRT_data_countries['C7M_Restrictions on internal movement'].max() + ((OxCGRT_data_countries['C7M_Flag'] == 1) * 0.5))
    OxCGRT_data_countries.loc[: , 'C8M'] = (100*(OxCGRT_data_countries['C8EV_International travel controls'])) / OxCGRT_data_countries['C8EV_International travel controls'].max()
    OxCGRT_data_countries.loc[: , 'Containment_Index'] = (OxCGRT_data_countries['C1M'] + OxCGRT_data_countries['C2M'] + OxCGRT_data_countries['C3M'] + 
                                                   OxCGRT_data_countries['C4M'] + OxCGRT_data_countries['C5M'] + OxCGRT_data_countries['C6M'] + OxCGRT_data_countries['C7M'] + OxCGRT_data_countries['C8M']) / 8
    strictness = OxCGRT_data_countries[['CountryName','Date','Containment_Index']]

    return strictness


def process_Google_mobility(Global_mobility):
    global_mob_process=Global_mobility.query('country_region_code=="FR"|country_region_code=="DE"|country_region_code=="NL"|country_region_code=="RS"|country_region_code=="ES"|country_region_code=="FI"')
    global_mob_process.loc[:,'date'] = pd.to_datetime(global_mob_process['date'])
    global_mob_process.columns= ['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2',
     'metro_area', 'iso_3166_2_code', 'census_fips_code', 'place_id',
     'date','Retail & Recreation',
           'Grocery & Pharmacy',
           'Parks',
           'Transit Stations',
           'Workplaces',
           'Residential'
    ]
    global_mob_process.set_index('date',inplace =True)
    google_columns = ['Retail & Recreation', 'Grocery & Pharmacy', 'Parks', 'Transit Stations', 'Workplaces', 'Residential']
    global_mob_process_de=global_mob_process[global_mob_process['country_region_code']=="DE"][google_columns]
    global_mob_process_fr=global_mob_process[global_mob_process['country_region_code']=="FR"][google_columns]
    global_mob_process_nl=global_mob_process[global_mob_process['country_region_code']=="NL"][google_columns]
    global_mob_process_rs=global_mob_process[global_mob_process['country_region_code']=="RS"][google_columns]
    global_mob_process_es=global_mob_process[global_mob_process['country_region_code']=="ES"][google_columns]
    global_mob_process_fi=global_mob_process[global_mob_process['country_region_code']=="FI"][google_columns]
    google_mobility_dict = {'Finland': global_mob_process_fi,
                   'France': global_mob_process_fr,
                   'Germany': global_mob_process_de,
                   'Netherlands': global_mob_process_nl,
                   'Serbia': global_mob_process_rs,
                   'Spain': global_mob_process_es
                  }
    return google_mobility_dict



def get_apple_countries_pivot(apple_mobility):
    apple_mob_process=apple_mobility.query('region=="France"|region=="Germany"|region=="Netherlands"|region=="Serbia"|region=="Spain"|region=="Finland"')
    apple_mob_process.set_index(['region', 'transportation_type', 'sub-region', 'country', 'geo_type', 'alternative_name'], inplace=True)
    apple_mob_process_stacked = apple_mob_process.stack()
    apple_countries = apple_mob_process_stacked.reset_index()
    apple_countries.columns = ['region', 'transportation_type', 'sub-region', 'country', 'geo_type', 'alternative_name', 'date', 'value']
    apple_countries['date'] = pd.to_datetime(apple_countries['date'], format='%Y-%m-%d')
    apple_countries_pivot = apple_countries.pivot_table(index=['date'], columns=['region','transportation_type'], values='value')
    return apple_countries_pivot



def get_20_21_energy(countries):
    countries_20_21 = []
    for country in countries:
        countries_20_21.append(pd.concat([country.loc['2020'].copy(),country.loc['2021'].copy()]))
    return countries_20_21


def get_percent(df):
    new_df = df[['Renewable_Percent']].copy()
    for resource in [ 'Coal', 'Oil', 'Gas', 'Biomass', 'Geothermal', 'Hydro', 'Nuclear', 'Solar', 'Waste', 'Wind']:
        new_df.loc[:,resource] = (df[resource]/df['Total'])
    for time in ['Month', 'DayofMonth', 'DayofWeek', 'Year']:
         new_df.loc[:,time] = df[time]
    return new_df

