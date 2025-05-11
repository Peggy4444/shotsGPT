import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric

from classes.data_point import Player, Country
from classes.data_source import PlayerStats, CountryStats
from typing import Union
from classes.data_point import Player
from classes.data_source import PlayerStats
import utils.constants as const



def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    # color: hexadecimal
    # alpha: transparency value between 0 and 1 (default is 1.0, fully opaque)
    s = (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )
    return s


class Visual:
    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")

    def __init__(self, pdf=False, plot_type="scout"):
        self.pdf = pdf
        if pdf:
            self.font_size_multiplier = 1.4
        else:
            self.font_size_multiplier = 1.0
        self.fig = go.Figure()
        self._setup_styles()

        if plot_type == "scout":
            self.annotation_text = (
                "<span style=''>{metric_name}: {data:.2f}</span>"
            )
        else:
            self.annotation_text = "<span style=''>{metric_name}: {data:.2f}</span>"

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def close(self):
        pass

    def _setup_styles(self):
        side_margin = 60
        top_margin = 75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.white),
            plot_bgcolor=rgb_to_color(self.white),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.dark_green),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",  # Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.dark_green, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title, subtitle):
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.dark_green),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text):
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.dark_green, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )


class DistributionPlot(Visual):
    def __init__(self, columns, labels=None, annotate=True, row_distance=1., *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.annotate = annotate
        self.row_distance = row_distance
        self.marker_color = (
            c for c in [Visual.dark_green, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _get_x_range(self):
        """
        Determine the minimum and maximum x-values across all traces in the figure.
        """
        x_values = []
        for trace in self.fig.data:
            if 'x' in trace:  # Check if the trace has x-values
                x_values.extend(trace['x'])  # Append all x-values from the trace

        # Return the min and max, or use defaults if no data is present
        #return (min(x_values) if x_values else -7, max(x_values) if x_values else 7)
        return (-1,1)


    def _setup_axes(self, labels=["Negative", "Average Contribution to xT", "Positive"]):

        x_min, x_max = self._get_x_range()  # Function to calculate min and max x values
        dynamic_width = max(100, (x_max - x_min) * 100)

        self.fig.update_layout(
            autosize=False,
            width=dynamic_width,  # Set figure width dynamically
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize margins
    )
        self.fig.update_xaxes(
            range=[x_min, x_max],
            fixedrange=True,
            tickmode="array",
            tickvals=[(x_min + x_max) / 2 - 3, (x_min + x_max) / 2, (x_min + x_max) / 2 + 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True
        x_min, x_max = self._get_x_range()

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            # **NEW: Add a horizontal line for each row**
            self.fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[i, i],  # Fixed y position for each row
                    mode="lines",
                    line=dict(color="gray", width=1, dash=None),  # Line style
                    showlegend=False,  # Hide from legend
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            y_pos = i * self.row_distance

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[y_pos],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            # Add annotations only if the flag is enabled
            if self.annotate:

                self.fig.add_annotation(
                    x=0,
                    y= y_pos + 0.4,
                    text=self.annotation_text.format(
                        metric_name=metric_name, data=ser_plot[col]
                    ),
                    showarrow=False,
                    font={
                        "color": rgb_to_color(self.dark_green),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )

    # def add_player(self, player: Player, n_group,metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_data_point(
    #         ser_plot=player.ser_metrics,
    #         plots = '_Z',
    #         name=player.name,
    #         hover='_Ranks',
    #         hover_string="Rank: %{customdata}/" + str(n_group)
    #     )

    def add_player(self, player: Union[Player, Country], n_group, metrics):

        # # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(player, Player):
            ser_plot = player.ser_metrics
            name = player.name
        elif isinstance(player, Country):  # Adjust this based on your class structure
            ser_plot = (
                player.ser_metrics
            )  # Assuming countries have a similar metric structure
            name = player.name
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )


    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    # def add_title_from_player(self, player: Player):
    #     self.player = player

    #     title = f"Evaluation of {player.name}?"
    #     subtitle = f"Based on {player.minutes_played} minutes played"

    #     self.add_title(title, subtitle)

    def add_title_from_player(self, player: Union[Player, Country]):
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)

class ShotContributionPlot1(DistributionPlot):
    def __init__(self, df_contributions, metrics, **kwargs):
        """
        Parameters:
        - df_contributions: DataFrame of contributions (rows: shots, columns: contributions).
        - metrics: List of metrics (columns in df_contributions) to plot.
        """
        self.df_contributions = df_contributions
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, **kwargs)

    def _setup_axes(self):
        """Set up axes for the distribution plot."""
        self.fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(self.columns))),  # One tick per feature
            ticktext=[format_metric(col) for col in self.columns],  # Use formatted metric names
            title="Features",
            showgrid=False,
        )

        self.fig.update_xaxes(
            #title="Contribution Value",
            title="",
            showgrid=False,
        )

    def add_shot(self, contribution_df, shot_id, metrics , id_to_number):
        """
        Add a single individual's contributions to the plot.
        """
        filtered_df = contribution_df[contribution_df["id"] == shot_id]
        if filtered_df.empty:
            raise ValueError(f"Shot ID {shot_id} not found in the contribution DataFrame.")
        if len(filtered_df) > 1:
            raise ValueError(f"Multiple rows found for Shot ID {shot_id}. Ensure IDs are unique.")
        contributions = filtered_df.iloc[0][metrics]
        

        self.add_data_point(
            ser_plot=contributions,  # This should now be a Series with contributions for the metrics
            plots="",
            name=f"Shot #{id_to_number[shot_id]}",  # Use the shot ID as the label
            hover="",
            hover_string="Value: %{customdata:.2f}",
        )

    def add_shots(self, df_shots, metrics):
        """
        Add contributions for all shots to the plot.
        """
        self.add_group_data(
            df_plot=self.df_contributions,
            plots="",  # Use the original column names
            names=df_shots["id"].astype(str),  # Shot IDs for hover text
            hover="",
            hover_string="Value: %{customdata:.2f}",
            legend="All Shots",
        )


class ShotContributionPlot(DistributionPlot):
    def __init__(self, df_contributions, df_shots, metrics, **kwargs):
        """
        Parameters:
        - df_contributions: DataFrame of contributions (rows: shots, columns: contributions).
        - metrics: List of metrics (columns in df_contributions) to plot.
        """
        self.df_contributions = df_contributions
        self.df_shots = df_shots
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)

    

    def add_shot(self, contribution_df, shots_df, shot_id, metrics, id_to_number):
        """
        Add a single shot to the plot with hover text displaying all feature values.
        """
        # Filter contributions and features for the selected shot
        filtered_contrib = contribution_df[contribution_df["id"] == shot_id]
        filtered_shot = shots_df[shots_df["id"] == shot_id]

        if filtered_contrib.empty or filtered_shot.empty:
            raise ValueError(f"Shot ID {shot_id} not found in the provided DataFrames.")
        if len(filtered_contrib) > 1 or len(filtered_shot) > 1:
            raise ValueError(f"Multiple rows found for Shot ID {shot_id}. Ensure IDs are unique.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_shot.iloc[0][feature_columns]
        player= filtered_shot['player_name'].values[0]

        # Generate hover text
        hover_text = [f"Player: {player}"]
        hover_text.append(f"Shot ID: {shot_id}")
        binary_features = [
            "shot_during_regular_play",
            "shot_with_left_foot",
            "shot_after_throw_in",
            "shot_after_corner",
            "shot_after_free_kick",
            "header",
        ]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]  # Use feature_column to avoid KeyError
            # Check if feature value is binary (0 or 1) and modify hover text accordingly
            if feature_column in binary_features:
                feature_text = "Yes" if feature_value == 1 else "No" if feature_value == 0 else f"{feature_value:.2f}"
            else:
                feature_text = f"{feature_value:.2f}"
            hover_text.append(f"{format_metric(feature_column)}: {feature_text}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Shot #{id_to_number[shot_id]}",
            hover="",  # Not using an additional column for hover values
            hover_string="<br>".join(hover_text),  # Combine hover text into a single string
        )

        # Annotate with feature names and values
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]  # Use feature_column to avoid KeyError
            # Check if feature value is binary (0 or 1) and modify annotation accordingly
            if feature_column in binary_features:
                feature_text = "Yes" if feature_value == 1 else "No" if feature_value == 0 else f"{feature_value:.2f}"
            else:
                feature_text = f"{feature_value:.2f}"

            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_text}",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.dark_green),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
                align="center",
            )


    def add_shots(self, df_shots, metrics, id_to_number):
        """
        Add a distribution plot for all shots, with hover text showing feature values for each metric.
        """
        # Prepare hover text for all metrics
        hover_texts = []
        
        for _, row in self.df_contributions.iterrows():
            hover_text = []
            shot_id = row["id"]  # Assuming 'id' is a shared identifier in both DataFrames
            shot_number = id_to_number.get(shot_id, "Unknown")
            player= df_shots[df_shots['id']==shot_id]['player_name'].values[0]
            hover_text.append(f"Player: {player}")
            hover_text.append(f"Shot #{shot_number}")

            # Retrieve the matching shot features
            shot_features = df_shots[df_shots["id"] == shot_id]

            if not shot_features.empty:
                shot_features = shot_features.iloc[0]
                for metric in metrics:
                    # Get the corresponding feature name
                    feature_column = metric.replace("_contribution", "")

                    if feature_column in shot_features:
                        # Format hover text for each feature
                        feature_value = shot_features[feature_column]
                        feature_text = (
                            "Yes" if feature_value == 1 else
                            "No" if feature_value == 0 else
                            f"{feature_value:.2f}"
                        )
                        hover_text.append(f"{format_metric(feature_column)}: {feature_text}")
                    else:
                        hover_text.append(f"{format_metric(feature_column)}: N/A")
            else:
                hover_text.append("No matching shot data")

            hover_texts.append("<br>".join(hover_text))

        # Add the group data to the plot
        self.add_group_data(
            df_plot=self.df_contributions,
            plots="",  # Use the original column names
            names=hover_texts,  # Include hover text for each metric
            hover="",  # No extra hover column suffix
            hover_string="",  # Hover text format already included
            legend="All Shots",
        )


class PassContributionPlot_Logistic(DistributionPlot):
    def __init__(self, df_contributions, df_passes, metrics, **kwargs):
        self.df_contributions = df_contributions
        self.df_passes = df_passes
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)
    
    def _get_x_range(self):
        x_values = []
        for trace in self.fig.data:
            if hasattr(trace, 'x') and trace['x'] is not None:
                x_values.extend(trace['x'])

        if x_values:
            min_x, max_x = min(x_values), max(x_values)
            padding = 0.1 * (max_x - min_x) if max_x != min_x else 1
            return min_x - padding, max_x + padding
        else:
            return -1, 1


    def add_pass(self, contribution_df, pass_df, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = contribution_df[contribution_df["id"] == pass_id]
        filtered_pass = pass_df[pass_df["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self, df_passes, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.df_contributions.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = df_passes[df_passes["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_contributions,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )


class xnn_plot(Visual):
    def __init__(self, columns, labels=None, annotate=True, row_distance=1.0, *args, **kwargs):
        self.columns = columns
        self.annotate = annotate
        self.row_distance = row_distance
        self.marker_color = (c for c in [Visual.dark_green, Visual.bright_yellow, Visual.bright_blue])
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)


    def finalize_axes(self, labels=["Negative", "Average Contribution to xT", "Positive"],x_min=-1, x_max=1):
        dynamic_width = max(100, (x_max - x_min) * 100)

        self.fig.update_layout(
            autosize=False,
            width=dynamic_width,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        self.fig.update_xaxes(
            range=[x_min, x_max],
            fixedrange=True,
            tickmode="array",
            tickvals=[x_min, 0, x_max],  # Centered around 0
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def draw_reference_lines(self,x_min,x_max):
        for i in range(len(self.columns)):
            y_val = i * self.row_distance
            self.fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=y_val,
                y1=y_val,
                line=dict(color="gray", width=1),
                xref="x",
                yref="y",
            )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True
        for i, col in enumerate(self.columns):
            y_val = i * self.row_distance
            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * y_val,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(self, ser_plot, plots, name, hover="", hover_string="", text=None):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            y_pos = i * self.row_distance
            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[y_pos],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            if self.annotate:
                self.fig.add_annotation(
                    x=0,
                    y=y_pos + 0.4,
                    text=self.annotation_text.format(
                        metric_name=format_metric(col), data=ser_plot[col]
                    ),
                    showarrow=False,
                    font={
                        "color": rgb_to_color(self.dark_green),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )



class PassContributionPlot_Xnn(xnn_plot):
    def __init__(self, df_xnn_contrib , df_passes_xnn , metrics, **kwargs):
        self.df_xnn_contrib = df_xnn_contrib
        self.df_passes_xnn = df_passes_xnn
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_xnn_contrib.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)

    def add_pass(self, df_xnn_contrib , df_passes_xnn, pass_id, metrics, selected_pass_id):

        #scale_factor = 100 
        # Filter contributions and features for the selected pass
        filtered_contrib = df_xnn_contrib[df_xnn_contrib["id"] == pass_id]
        filtered_pass = df_passes_xnn[df_passes_xnn["id"] == pass_id]

        feature_columns = [m.replace("_contribution", "") for m in metrics]
        contributions = filtered_contrib.iloc[0][metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )
         # --- Dynamic Scaling ---
        all_contributions = pd.concat([self.df_xnn_contrib[self.metrics], contributions.to_frame().T])

        min_val = float(all_contributions.min().min())
        max_val = float(all_contributions.max().max())
        buffer = 0.25 * max(abs(min_val), abs(max_val))

        x_min = -(max(abs(min_val), abs(max_val)) + buffer)
        x_max = (max(abs(min_val), abs(max_val)) + buffer)

         # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
                },
            align="center",
            )
        

        self.draw_reference_lines(x_min, x_max)
        self.finalize_axes(x_min=x_min, x_max=x_max)

    def add_passes(self, df_passes_xnn , metrics):
        hover_texts = []

        for _, row in self.df_xnn_contrib.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = df_passes_xnn[df_passes_xnn["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_xnn_contrib,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )
        # Add jittered dots for all passes
        for i, metric in enumerate(metrics):
            y = np.full(len(self.df_xnn_contrib), i) + np.random.normal(0, 0.12, len(self.df_xnn_contrib))
            x = self.df_xnn_contrib[metric]

            self.fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=6, color='rgba(0, 128, 0, 0.4)'),  # Green with transparency
                hoverinfo="skip",
                showlegend=False
            ))

            
            # Final axis style and layout tweaks
            self.fig.update_xaxes(
            showgrid=False,
            zeroline=True,
            zerolinecolor='lightgray',
            #range=x_range,  # adjust range to fit your data
            title_font=dict(size=14, family="Arial")
        )

        self.fig.update_yaxes(
            tickfont=dict(size=12, family="Arial")
        )




class DistributionPlot_xnn_models(Visual):
    def __init__(self, columns, labels=None, annotate=True, row_distance=1.0, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.annotate = annotate
        self.row_distance = row_distance
        self.marker_color = (
            c for c in [Visual.dark_green, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _get_x_range(self):
        """
        Determine the minimum and maximum x-values across all traces in the figure.
        """
        x_values = []
        for trace in self.fig.data:
            if 'x' in trace:  # Check if the trace has x-values
                x_values.extend(trace['x'])  # Append all x-values from the trace

        # Return the min and max, or use defaults if no data is present
        #return (min(x_values) if x_values else -7, max(x_values) if x_values else 7)
        return (-1,1)


    def _setup_axes(self, labels=["Negative", "Average Contribution to xT", "Positive"]):

        x_min, x_max = self._get_x_range()  # Function to calculate min and max x values
        dynamic_width = max(100, (x_max - x_min) * 100)

        self.fig.update_layout(
            autosize=False,
            width=dynamic_width,  # Set figure width dynamically
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize margins
    )
        self.fig.update_xaxes(
            range=[x_min, x_max],
            fixedrange=True,
            tickmode="array",
            tickvals=[(x_min + x_max) / 2 - 3, (x_min + x_max) / 2, (x_min + x_max) / 2 + 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
            showgrid = False,
            zeroline = False
        )

        lines = []
        for i in range(len(self.columns)):
            lines.append(dict(
                type="line",
                x0=x_min, x1=x_max,
                y0=i*self.row_distance, y1=i*self.row_distance,
                xref="x", yref="y",
                line=dict(color=rgb_to_color(self.medium_green), width=1)
            ))
        self.fig.update_layout(shapes=lines)

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string
            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            y_pos = i * self.row_distance

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[y_pos],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            # Add annotations only if the flag is enabled
            if self.annotate:

                self.fig.add_annotation(
                    x=ser_plot[col + plots],
                    y= y_pos,
                    xanchor="center",
                    yanchor="bottom",
                    text=self.annotation_text.format(
                        metric_name=metric_name, data=ser_plot[col]
                    ),
                    showarrow=False,
                    font={
                        "color": rgb_to_color(self.dark_green),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )

class ModelContributionPlotXNNShap(DistributionPlot_xnn_models):
    def __init__(self, shap_df_xnn, pass_df_xnn, metrics_shap, **kwargs):
        self.shap_df_xnn = shap_df_xnn
        self.pass_df_xnn = pass_df_xnn
        self.metrics_shap = metrics_shap
 # Validate inputs
        for metric in metrics_shap:
            if metric not in shap_df_xnn.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics_shap, annotate=False, **kwargs)  

    def add_pass(self, shap_df_xnn, pass_df_xnn, pass_id, metrics_shap, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib_xnn = shap_df_xnn[shap_df_xnn["id"] == pass_id]
        filtered_pass_xnn = pass_df_xnn[pass_df_xnn["id"] == pass_id]

        if filtered_contrib_xnn.empty or filtered_pass_xnn.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib_xnn) > 1 or len(filtered_pass_xnn) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib_xnn.iloc[0][metrics_shap]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics_shap]
        feature_values = filtered_pass_xnn.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics_shap, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                #y=i * 1.5 + 0.5,  # or 2.0 for more spacing

                y=i * 1.0 + 0.2,
                xanchor="center",
                yanchor="bottom",
                yshift=2,
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 11 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self,pass_df_xnn, metrics_shap, selected_pass_id):
        hover_texts = []

        for _, row in self.shap_df_xnn.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = pass_df_xnn[pass_df_xnn["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics_shap:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.shap_df_xnn,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )



 ### contribution plot of logistic pressure based model 
class PassContributionPlot_Logistic_pressure(DistributionPlot_xnn_models):
    def __init__(self, df_contributions_pressure, df_passes, metrics, **kwargs):
        self.df_contributions_pressure = df_contributions_pressure
        self.df_passes = df_passes
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions_pressure.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)
    
    def _get_x_range(self):
        x_values = []
        for trace in self.fig.data:
            if hasattr(trace, 'x') and trace['x'] is not None:
                x_values.extend(trace['x'])

        if x_values:
            min_x, max_x = min(x_values), max(x_values)
            padding = 0.1 * (max_x - min_x) if max_x != min_x else 1
            return min_x - padding, max_x + padding
        else:
            return -1, 1


    def add_pass(self, contribution_df_pressure, pass_df, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = contribution_df_pressure[contribution_df_pressure["id"] == pass_id]
        filtered_pass = pass_df[pass_df["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self, df_passes, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.df_contributions_pressure.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = df_passes[df_passes["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_contributions_pressure,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )


class DistributionPlot_speed_models(Visual):
    def __init__(self, columns, labels=None, annotate=True, row_distance=1.0, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.annotate = annotate
        self.row_distance = row_distance
        self.marker_color = (
            c for c in [Visual.dark_green, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _get_x_range(self):
        """
        Determine the minimum and maximum x-values across all traces in the figure.
        """
        x_values = []
        for trace in self.fig.data:
            if 'x' in trace:  # Check if the trace has x-values
                x_values.extend(trace['x'])  # Append all x-values from the trace

        # Return the min and max, or use defaults if no data is present
        #return (min(x_values) if x_values else -7, max(x_values) if x_values else 7)
        return (-1,1)


    def _setup_axes(self, labels=["Negative", "Average Contribution to xT", "Positive"]):

        x_min, x_max = self._get_x_range()  # Function to calculate min and max x values
        dynamic_width = max(100, (x_max - x_min) * 100)

        self.fig.update_layout(
            autosize=False,
            width=dynamic_width,  # Set figure width dynamically
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize margins
    )
        self.fig.update_xaxes(
            range=[x_min, x_max],
            fixedrange=True,
            tickmode="array",
            tickvals=[(x_min + x_max) / 2 - 3, (x_min + x_max) / 2, (x_min + x_max) / 2 + 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
            showgrid = False,
            zeroline = False
        )

        lines = []
        for i in range(len(self.columns)):
            lines.append(dict(
                type="line",
                x0=x_min, x1=x_max,
                y0=i*self.row_distance, y1=i*self.row_distance,
                xref="x", yref="y",
                line=dict(color=rgb_to_color(self.medium_green), width=1)
            ))
        self.fig.update_layout(shapes=lines)

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string
            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            y_pos = i * self.row_distance


            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[y_pos],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            # Add annotations only if the flag is enabled
            if self.annotate:

                self.fig.add_annotation(
                    x=ser_plot[col + plots],
                    y= y_pos,
                    xanchor="center",
                    yanchor="bottom",
                    text=self.annotation_text.format(
                        metric_name=metric_name, data=ser_plot[col]
                    ),
                    showarrow=False,
                    font={
                        "color": rgb_to_color(self.dark_green),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )


### distribution plot for speed based models 
class PassContributionPlot_Logistic_speed(DistributionPlot_speed_models):
    def __init__(self, df_contributions_speed, df_passes, metrics, **kwargs):
        self.df_contributions_speed = df_contributions_speed
        self.df_passes = df_passes
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions_speed.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)
    
    def _get_x_range(self):
        x_values = []
        for trace in self.fig.data:
            if hasattr(trace, 'x') and trace['x'] is not None:
                x_values.extend(trace['x'])

        if x_values:
            min_x, max_x = min(x_values), max(x_values)
            padding = 0.1 * (max_x - min_x) if max_x != min_x else 1
            return min_x - padding, max_x + padding
        else:
            return -1, 1


    def add_pass(self, contribution_df_speed, pass_df, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = contribution_df_speed[contribution_df_speed["id"] == pass_id]
        filtered_pass = pass_df[pass_df["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1.0 + 0.2,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self, df_passes, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.df_contributions_speed.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = df_passes[df_passes["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_contributions_speed,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )


class DistributionPlot_position_model(Visual):
    def __init__(self, columns, labels=None, annotate=True, row_distance=1.0, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.annotate = annotate
        self.row_distance = row_distance
        self.marker_color = (
            c for c in [Visual.dark_green, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _get_x_range(self):
        x_values = []
        for trace in self.fig.data:
            if 'x' in trace and trace['x'] is not None:
                x_values.extend(trace['x'])
        if not x_values:
            # fallback
            return -1, 1

        min_x, max_x = min(x_values), max(x_values)
        padding = 0.1 * (max_x - min_x) if max_x > min_x else 1
        return min_x - padding, max_x + padding


    def _setup_axes(self, labels=["Negative", "Average Contribution to xT", "Positive"]):

        x_min, x_max = self._get_x_range()  # Function to calculate min and max x values
        dynamic_width = max(100, (x_max - x_min) * 100)

        self.fig.update_layout(
            autosize=False,
            width=dynamic_width,  # Set figure width dynamically
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize margins
    )
        self.fig.update_xaxes(
            range=[x_min, x_max],
            fixedrange=True,
            tickmode="array",
            tickvals=[(x_min + x_max) / 2 - 3, (x_min + x_max) / 2, (x_min + x_max) / 2 + 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
            showgrid = False,
            zeroline = False
        )

        lines = []
        for i in range(len(self.columns)):
            lines.append(dict(
                type="line",
                x0=x_min, x1=x_max,
                y0=i*self.row_distance, y1=i*self.row_distance,
                xref="x", yref="y",
                line=dict(color=rgb_to_color(self.medium_green), width=1)
            ))
        self.fig.update_layout(shapes=lines)

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string
            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            y_pos = i * self.row_distance

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[y_pos],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            # Add annotations only if the flag is enabled
            if self.annotate:

                self.fig.add_annotation(
                    x=ser_plot[col + plots],
                    y= y_pos,
                    xanchor="center",
                    yanchor="bottom",
                    text=self.annotation_text.format(
                        metric_name=metric_name, data=ser_plot[col]
                    ),
                    showarrow=False,
                    font={
                        "color": rgb_to_color(self.dark_green),
                        "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier,
                    },
                )


#position based model distribution plot 
class PassContributionPlot_Logistic_position(DistributionPlot_position_model):
    def __init__(self, df_contributions_position, df_passes, metrics, **kwargs):
        self.df_contributions_position = df_contributions_position
        self.df_passes = df_passes
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions_position.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)
    
    def _get_x_range(self):
        x_values = []
        for trace in self.fig.data:
            if hasattr(trace, 'x') and trace['x'] is not None:
                x_values.extend(trace['x'])

        if x_values:
            min_x, max_x = min(x_values), max(x_values)
            padding = 0.1 * (max_x - min_x) if max_x != min_x else 1
            return min_x - padding, max_x + padding
        else:
            return -1, 1


    def add_pass(self, contribution_df_position, pass_df, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = contribution_df_position[contribution_df_position["id"] == pass_id]
        filtered_pass = pass_df[pass_df["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            val = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i + 0.5,
                xref="x",
                yref="y",
                text=f"{format_metric(feature_column)}: {val:.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font={
                    "color": rgb_to_color(self.dark_green),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

        # …then once, stretch out the y‐axis so nothing ever gets hidden:
        self.fig.update_yaxes(range=[-0.5, len(metrics) - 0.5])



    def add_passes(self, df_passes, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.df_contributions_position.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = df_passes[df_passes["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_contributions_position,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )

#event based model distribution plot 
class PassContributionPlot_Logistic_event(DistributionPlot_xnn_models):
    def __init__(self, df_contributions_event, df_passes, metrics, **kwargs):
        self.df_contributions_event = df_contributions_event
        self.df_passes = df_passes
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions_event.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)
    
    def _get_x_range(self):
        x_values = []
        for trace in self.fig.data:
            if hasattr(trace, 'x') and trace['x'] is not None:
                x_values.extend(trace['x'])

        if x_values:
            min_x, max_x = min(x_values), max(x_values)
            padding = 0.1 * (max_x - min_x) if max_x != min_x else 1
            return min_x - padding, max_x + padding
        else:
            return -1, 1


    def add_pass(self, contribution_df_event, pass_df, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = contribution_df_event[contribution_df_event["id"] == pass_id]
        filtered_pass = pass_df[pass_df["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self, df_passes, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.df_contributions_event.iterrows():
            hover_text = []
            pass_id = row["id"]
            #pass_number = selected_pass_id
            hover_text.append(f"Pass #{pass_id}")
            pass_features = df_passes[df_passes["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_contributions_event,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )


class PassContributionPlot_XGBoost(DistributionPlot):
    def __init__(self, feature_contrib_df, pass_df_xgboost, metrics, **kwargs):
        self.feature_contrib_df = feature_contrib_df
        self.pass_df_xgboost = pass_df_xgboost
        self.metrics = metrics
 # Validate inputs
        for metric in metrics:
            if metric not in feature_contrib_df.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)  

    def add_pass(self, feature_contrib_df, pass_df_xgboost, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = feature_contrib_df[feature_contrib_df["id"] == pass_id]
        filtered_pass = pass_df_xgboost[pass_df_xgboost["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                #y=i * 1.5 + 0.5,  # or 2.0 for more spacing

                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 11 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self, feature_contrib_df, pass_df_xgboost, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.feature_contrib_df.iterrows():
            hover_text = []
            pass_id = row["id"]
            pass_number = selected_pass_id
            #hover_text.append(f"Pass #{pass_number}")
            pass_features = pass_df_xgboost[pass_df_xgboost["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.feature_contrib_df,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )

    
        # Validate inputs
        for metric in metrics:
            if metric not in feature_contrib_df.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)  

    def add_pass(self, feature_contrib_df, pass_df_xgboost, pass_id, metrics, selected_pass_id):
        # Filter contributions and features for the selected pass
        filtered_contrib = feature_contrib_df[feature_contrib_df["id"] == pass_id]
        filtered_pass = pass_df_xgboost[pass_df_xgboost["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution", "") for metric in metrics]
        feature_values = filtered_pass.iloc[0][feature_columns]

        # Construct hover text
        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        # Add contributions to the plot
        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        # Annotate features
        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                #y=i * 1.5 + 0.5,  # or 2.0 for more spacing

                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                "color": rgb_to_color(self.dark_green),
                "family": "Gilroy-Light",
                "size": 11 * self.font_size_multiplier,
                },
            align="center",
            )



    def add_passes(self, pass_df_xgboost, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.feature_contrib_df.iterrows():
            hover_text = []
            pass_id = row["id"]
            pass_number = selected_pass_id
            #hover_text.append(f"Pass #{pass_number}")
            pass_features = pass_df_xgboost[pass_df_xgboost["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]

                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.feature_contrib_df,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )

class PassContributionPlot_Mimic(DistributionPlot):
    def __init__(self, df_contributions_mimic, df_passes, metrics, **kwargs):
        self.df_contributions = df_contributions_mimic
        self.df_passes = df_passes
        self.metrics = metrics

        # Validate inputs
        for metric in metrics:
            if metric not in df_contributions_mimic.columns:
                raise ValueError(f"Metric '{metric}' is not a column in df_contributions.")

        super().__init__(columns=metrics, annotate=False, **kwargs)

    def add_pass(self, contribution_df, pass_df, pass_id, metrics, selected_pass_id):
        filtered_contrib = contribution_df[contribution_df["id"] == pass_id]
        filtered_pass = pass_df[pass_df["id"] == pass_id]

        if filtered_contrib.empty or filtered_pass.empty:
            raise ValueError(f"Pass ID {pass_id} not found.")
        if len(filtered_contrib) > 1 or len(filtered_pass) > 1:
            raise ValueError(f"Multiple rows found for Pass ID {pass_id}.")

        contributions = filtered_contrib.iloc[0][metrics]
        feature_columns = [metric.replace("_contribution_mimic", "") for metric in metrics]

        feature_values = filtered_pass.iloc[0][feature_columns]

        hover_text = [f"Pass ID: {selected_pass_id}"]
        for feature_column in feature_columns:
            feature_value = feature_values[feature_column]
            hover_text.append(f"{format_metric(feature_column)}: {feature_value:.2f}")

        self.add_data_point(
            ser_plot=contributions,
            plots="",
            name=f"Pass #{selected_pass_id}",
            hover="",
            hover_string="<br>".join(hover_text)
        )

        for i, (metric, feature_column) in enumerate(zip(metrics, feature_columns)):
            feature_value = feature_values[feature_column]
            self.fig.add_annotation(
                x=contributions[metric],
                y=i * 1.0 + 0.5,
                xanchor="center",
                text=f"{format_metric(feature_column)}: {feature_value:.2f}",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.dark_green),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
                align="center",
            )

    def add_passes(self, df_passes, metrics, selected_pass_id):
        hover_texts = []

        for _, row in self.df_contributions.iterrows():
            hover_text = []
            pass_id = row["id"]
            pass_features = df_passes[df_passes["id"] == pass_id]
            if not pass_features.empty:
                pass_features = pass_features.iloc[0]
                for metric in metrics:
                    feature_column = metric.replace("_contribution", "")
                    if feature_column in pass_features:
                        value = pass_features[feature_column]
                        hover_text.append(f"{format_metric(feature_column)}: {value:.2f}")
            else:
                hover_text.append("No matching pass data")

            hover_texts.append("<br>".join(hover_text))

        self.add_group_data(
            df_plot=self.df_contributions,
            plots="",
            names=hover_texts,
            hover="",
            hover_string="",
            legend="All Passes",
        )


class PitchVisual(Visual):
    def __init__(self, metric, pdf = False, *args, **kwargs):
        self.metric = metric
        super().__init__(*args, **kwargs)
        self._add_pitch()
        self.pdf = pdf

    @staticmethod
    def ellipse_arc(x_center=0, y_center=0, a=1, b=1, start_angle=0, end_angle=2 * np.pi, N=100):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        return path

    def _add_pitch(self):
        self.fig.update_layout(
            hoverdistance=100,
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=True,
                zeroline=False,
                # Range slightly larger to avoid half of line being hidden by edge of plot
                range=[-0.2, 105],
                scaleanchor="y",
                scaleratio=1.544,
                constrain="domain",
                tickvals=[25, 75],
                ticktext=["Defensive", "Offensive"],
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                # Range slightly larger to avoid half of line being hidden by edge of plot
                range=[-0.2, 68],
                constrain="domain",
                fixedrange=True,
            ),
        )
        shapes = self._get_shapes()
        for shape in shapes:
            shape.update(dict(line={"color": "green", "width": 2}, xref="x", yref="y", ))
            self.fig.add_shape(**shape)

    def _get_shapes(self):
        # Plotly doesn't support arcs svg paths, so we need to create them manually
        shapes = [
            # Center circle
            dict(
                type="circle", x0=41.28, y0=36.54, x1=58.71, y1=63.46,
            ),
            # Own penalty area
            dict(
                type="rect", x0=0, y0=19, x1=16, y1=81,
            ),
            # Opponent penalty area
            dict(
                type="rect", x0=84, y0=19, x1=100, y1=81,
            ),
            dict(
                type="rect", x0=0, y0=0, x1=100, y1=100,
            ),
            # Halfway line
            dict(
                type="line", x0=50, y0=0, x1=50, y1=100,
            ),
            # Own goal area
            dict(
                type="rect", x0=0, y0=38, x1=6, y1=62,
            ),
            # Opponent goal area
            dict(
                type="rect", x0=94, y0=38, x1=100, y1=62,
            ),
            # Own penalty spot
            dict(
                type="circle", x0=11.2, y0=49.5, x1=11.8, y1=50.5, fillcolor="green"
            ),
            # Opponent penalty spot
            dict(
                type="circle", x0=89.2, y0=49.5, x1=89.8, y1=50.5, fillcolor="green"
            ),
            # Penalty arc
            # Not sure why we need to multiply the radii by 1.35, but it seems to work
            dict(
                type="path", path=self.ellipse_arc(11, 50, 6.2 * 1.35, 9.5 * 1.35, -0.3 * np.pi, 0.3 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(89, 50, 6.2 * 1.35, 9.5 * 1.35, 0.7 * np.pi, 1.3 * np.pi),
            ),
            # Corner arcs
            # Can't plot a part of a cirlce
            # dict(
            #     type="circle", x0=-6.2*0.3, y0=-9.5*0.3, x1=6.2*0.3, y1=9.5*0.3,
            # ),
            dict(
                type="path", path=self.ellipse_arc(0, 0, 6.2 * 0.3, 9.5 * 0.3, 0, np.pi / 2),
            ),
            dict(
                type="path", path=self.ellipse_arc(100, 0, 6.2 * 0.3, 9.5 * 0.3, np.pi / 2, np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(100, 100, 6.2 * 0.3, 9.5 * 0.3, np.pi, 3 / 2 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(0, 100, 6.2 * 0.3, 9.5 * 0.3, 3 / 2 * np.pi, 2 * np.pi),
            ),
            # Goals
            dict(
                type="rect", x0=-3, y0=44, x1=0, y1=56,
            ),
            dict(
                type="rect", x0=100, y0=44, x1=103, y1=56,
            ),
        ]
        return shapes

    def add_group_data(self, *args, **kwargs):
        pass

    def iter_zones(self, zone_dict=const.PITCH_ZONES_BBOX):
        for key, value in zone_dict.items():
            x = [
                value["x_lower_bound"],
                value["x_upper_bound"],
                value["x_upper_bound"],
                value["x_lower_bound"],
                value["x_lower_bound"],
            ]
            y = [
                value["y_lower_bound"],
                value["y_lower_bound"],
                value["y_upper_bound"],
                value["y_upper_bound"],
                value["y_lower_bound"],
            ]
            yield key, x, y

    def add_data_point(self, ser_plot, name, ser_hover, hover_string):
        for key, x, y in self.iter_zones():
            if key in ser_plot.index and ser_plot[key] == ser_plot[key]:
                self.fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line={"color": rgb_to_color(self.bright_green), "width": 1, },
                        fill="toself",
                        fillcolor=rgb_to_color(self.bright_green, opacity=float(ser_plot[key] / 100), ),
                        showlegend=False,
                        name=name,
                        hoverinfo="skip"
                    )
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=[x[0] / 2 + x[1] / 2],
                        y=[y[0] / 2 + y[2] / 2],
                        mode="text",
                        hovertemplate=name + '<br>Zone: ' + key.capitalize() + "<br>" + hover_string + '<extra></extra>',
                        text=describe_level(ser_hover[key][0]).capitalize(),
                        textposition="middle center",
                        textfont={"color": rgb_to_color(self.dark_green), "family": "Gilroy-Light",
                                  "size": 10 * self.font_size_multiplier},
                        customdata=[ser_hover[key]],
                        showlegend=False,
                    )
                )
            else:
                self.fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line={"width": 0, },
                        fill="toself",
                        fillcolor=rgb_to_color(self.gray, opacity=0.5),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=[x[0] / 2 + x[1] / 2],
                        y=[y[0] / 2 + y[2] / 2],
                        mode="none",
                        hovertemplate='Not enough data<extra></extra>',
                        text=[name],
                        showlegend=False,
                    )
                )

    def add_player(self, player, n_group, quality):
        metric = const.QUALITY_PITCH_METRICS[quality]
        multi_level = {}
        for key, _, _ in self.iter_zones():
            ser = player.ser_zones["Z"][metric]
            if key in ser.index and ser[key] == ser[key]:
                multi_level[key] = np.stack([
                    player.ser_zones["Z"][metric][key],
                    player.ser_zones["Raw"][metric][key]
                ], axis=-1)

        self.add_data_point(
            ser_plot=player.ser_zones["Rank_pct"][metric],
            ser_hover=multi_level,
            name=player.name,
            hover_string="Z-Score: %{customdata[0]:.2f}<br>%{customdata[1]:.2f} " + self.metric.lower(),
        )

    def add_title_from_player(self, player: Player, other_player: Player = None, quality=None):
        short_metric_name = self.metric.replace(" per 90", "").replace(" %", "")
        short_metric_name = short_metric_name[0].lower() + short_metric_name[1:]
        title = f"How is {player.name} at {'<i>' if not self.pdf else ''}{short_metric_name}{'</i>' if not self.pdf else ''}?"
        subtitle = f"{player.competition.get_plot_subtitle()} | {player.minutes} minutes played"
        self.add_title(title, subtitle)
        self.add_low_center_annotation(f"Compared to other {player.detailed_position.lower()}s")







class VerticalPitchVisual(PitchVisual):
    def _add_pitch(self):
        self.fig.update_layout(
            hoverdistance=100,
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.2, 100],
                constrain="domain",
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                range=[50, 100.2],
                scaleanchor="x",
                scaleratio=1.544,
                constrain="domain",
                fixedrange=True,
            ),
        )
        shapes = self._get_shapes()
        for shape in shapes:
            if shape["type"] != "path":
                shape.update(dict(line={"color": "green", "width": 2}, xref="x", yref="y", ))
                shape["x0"], shape["x1"], shape["y0"], shape["y1"] = shape["y0"], shape["y1"], shape["x0"], shape["x1"]
                self.fig.add_shape(**shape)

        # Add the arcs
        arcs = [
            dict(
                type="path", path=self.ellipse_arc(50, 89, 9.5 * 1.35, 6.2 * 1.35, -0.8 * np.pi, -0.2 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(100, 100, 9.5 * 0.3, 6.2 * 0.3, np.pi, 3 / 2 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(0, 100, 9.5 * 0.3, 6.2 * 0.3, 3 / 2 * np.pi, 2 * np.pi),
            )]

        for arc in arcs:
            arc.update(dict(line={"color": "green", "width": 2}, xref="x", yref="y", ))
            self.fig.add_shape(**arc)

    def iter_zones(self):
        for key, value in const.SHOT_ZONES_BBOX.items():
            x = [
                value["y_lower_bound"],
                value["y_upper_bound"],
                value["y_upper_bound"],
                value["y_lower_bound"],
                value["y_lower_bound"],
            ]
            y = [
                value["x_lower_bound"],
                value["x_lower_bound"],
                value["x_upper_bound"],
                value["x_upper_bound"],
                value["x_lower_bound"],
            ]
            yield key, x, y



class ShotVisual(VerticalPitchVisual):
    def __init__(self, *args, **kwargs):
        self.line_color = 'green'
        self.line_width = 3
        self.shot_color = rgb_to_color(self.bright_blue)
        self.failed_color = rgb_to_color(self.bright_yellow)
        self.bbox_color = rgb_to_color(self.bright_green)
        self.marker_size = 15
        self.basic_text_size = 10
        self.text_size = 20
        super().__init__(*args, **kwargs)

    def add_shots(self, shots):

        shots_df = shots.df_shots.copy()
        shot_contribution= shots.df_contributions.copy()    

        shots_df['category'] = shots_df['goal']
        labels = {False: 'Shot', True: 'Goal'}
        shots_df['category'] = shots_df['category'].replace(labels)
        arrays_to_stack = [shots_df.xG]

        # Stack the arrays
        customdata = np.stack(arrays_to_stack, axis=-1)

  
        shots_df['ms'] = shots_df['xG'] * 20 + 10

        masks = [shots_df['goal'] == False, shots_df['goal']== True]
        markers = [
            {"symbol": "circle-open", "color": rgb_to_color(self.dark_green, opacity=1),
             "line": {"color": rgb_to_color(self.dark_green, opacity=1), "width": 2}},
            {"symbol": "circle", "color": rgb_to_color(self.bright_green),
             "line": {"color": rgb_to_color(self.dark_green), "width": 2}}]

        names = ["Shot", "Goal"]
        filtered_data = [shots_df[mask] for mask in masks]
        temp_customdata = [customdata[mask] for mask in masks]

        for data, marker, name, custom in zip(filtered_data, markers, names, temp_customdata):
            if custom.size == 0:  # Skip if customdata is empty
                continue
            # hovertemplate = ('<b>%{customdata[3]}</b><br>%{customdata[0]}<br>Minute: %{customdata[1]}<br>'
            #                  '<b>xG:</b> %{customdata[2]:.3f}<br>')

            # feature_names = ['Angle', 'Header', 'Match State', 'Strong Foot', 'Smart Pass', 'Cross',
            #                  'Counterattack', 'Clear Header', 'Rebound', 'Key Pass', 'Self Created Shot']
            # for i, feature_name in enumerate(feature_names, start=4):
            #     hovertemplate += f'<b>{feature_name}:</b> %{{customdata[{i}]:.3f}}<br>' if custom[0][i] > 0.005 else ''

            #hovertemplate += '<extra></extra>'
            self.fig.add_trace(
                go.Scatter(
                    x=(68 - data['start_y'])*100/68, y=data['start_x']*100/105,
                    mode="markers",
                    #marker=marker,
                    marker=dict(size=10),
                    #marker_size=data['ms'],
                    showlegend=False,
                    name=name,
                    customdata=custom,
                    #hovertemplate=hovertemplate,
                )
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[-100],
                    y=[0],
                    mode="markers",
                    marker=marker,
                    name=name,
                    showlegend=True,
                    marker_size=10
                )
            )

        self.fig.update_layout(
            legend=dict(
                y=-0.05,
            )
        )

    def add_shot(self, shots, shot_id):
        # Filter for the specific shot using the shot_id
        shot_data = shots.df_shots[shots.df_shots['id'] == shot_id]
        if shot_data.empty:
            raise ValueError(f"Shot with ID {shot_id} not found.")

        # Extract the shot coordinates (start_x, start_y)
        shot_x = shot_data['start_x'].values[0]
        shot_y = shot_data['start_y'].values[0]
        end_x = shot_data['end_x'].values[0]
        end_y = shot_data['end_y'].values[0]
        goal_status = shot_data['goal'].values[0]  # True if goal, False if no goal
        xG_value = shot_data['xG'].values[0]  # Get the xG value
        player = shot_data['player_name'].values[0]
        team = shot_data['team_name'].values[0]
        # Add existing logic to plot shots here...
        shot_data['category'] = shot_data['goal']
        labels = {False: 'Shot', True: 'Goal'}
        shot_data['category'] = shot_data['category'].replace(labels)

        start_x_norm = shot_x * 100 / 105
        start_y_norm = (68 - shot_y) * 100 / 68
        end_x_norm = 100 - (end_x * 100 / 105)
        end_y_norm = (68 - end_y) * 100 / 68


        # Extract teammate coordinates (e.g., teammate_1_x, teammate_1_y, etc.)
        teammate_x_cols = [col for col in shot_data.columns if 'teammate' in col and '_x' in col]
        teammate_y_cols = [col for col in shot_data.columns if 'teammate' in col and '_y' in col]

        teammate_x = shot_data[teammate_x_cols].values[0]
        teammate_y = shot_data[teammate_y_cols].values[0]

        # Extract opponent coordinates (e.g., opponent_1_x, opponent_1_y, etc.)
        opponent_x_cols = [col for col in shot_data.columns if 'opponent' in col and '_x' in col]
        opponent_y_cols = [col for col in shot_data.columns if 'opponent' in col and '_y' in col]

        opponent_x = shot_data[opponent_x_cols].values[0]
        opponent_y = shot_data[opponent_y_cols].values[0]

        # Plot the shot location (start_x, start_y)
        self.fig.add_trace(
                            go.Scatter(
                                x=[start_y_norm],
                                y=[start_x_norm],
                                mode="markers",
                                marker=dict(size=12, color=self.shot_color, symbol="circle"),
                                name="Shot Start",
                                showlegend=True
                            )
                        )
        
        # Add an invisible scatter trace for the legend
        self.fig.add_trace(
            go.Scatter(
                x=[None],  # Invisible marker
                y=[None],
                mode="lines+markers",
                line=dict(color="green", width=2),  # Line matching the arrow style
                marker=dict(size=10, color="green", symbol="arrow-bar-up"),  # Arrow symbol
                name="Shot Direction",  # Legend label
                showlegend=True
            )
        )

        # Add shot arrow

        self.fig.add_annotation(
            x=end_y_norm,
            y=end_x_norm,
            ax=start_y_norm,
            ay=start_x_norm,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=2,  # Type of arrowhead
            arrowsize=1.5,  # Scale of the arrow
            arrowwidth=2,  # Width of the arrow line
            arrowcolor="green",  # Color of the arrow
            showarrow=True
        )



        # Plot teammates' locations
        self.fig.add_trace(
            go.Scatter(
                x=(68 - teammate_y) * 100 / 68,
                y=teammate_x * 100 / 105,
                mode="markers",
                marker=dict(size=10, color='blue', symbol="circle-open"),
                name="Teammates",
                showlegend=True
            )
        )

        # Plot opponents' locations
        self.fig.add_trace(
            go.Scatter(
                x=(68 - opponent_y) * 100 / 68,
                y=opponent_x * 100 / 105,
                mode="markers",
                marker=dict(size=10, color='red', symbol="x"),
                name="Opponents",
                showlegend=True
            )
        )

        self.fig.update_layout(
            title={
                'text': f"Shot by {player} from {team} | Outcome: {'Goal' if goal_status else 'No Goal'} | xG: {xG_value:.2f}",
                'font': {'color': 'green'}  # Set the title color to white
            },
            legend=dict(y=-0.05),
)

class HorizontalPitchVisual(PitchVisual):
    def _add_pitch(self):
        self.fig.update_layout(
            hoverdistance=100,
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.2, 100],  # Full width
                constrain="domain",
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.2, 100],  # Full height
                scaleanchor="x",
                scaleratio=0.65,  # approx 68/105
                constrain="domain",
                fixedrange=True,
            ),
        )

        shapes = self._get_shapes()
        for shape in shapes:
            if shape["type"] != "path":
                shape.update(dict(line={"color": "green", "width": 2}, xref="x", yref="y"))
                # In horizontal, use original x0, x1, y0, y1 — no need to swap
                self.fig.add_shape(**shape)

        # Add arcs for horizontal layout (adjust positions)
        arcs = [
            dict(
                type="path", path=self.ellipse_arc(11, 50, 6.2 * 1.35, 9.5 * 1.35, 1.2 * np.pi, 1.8 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(0, 0, 6.2 * 0.3, 9.5 * 0.3, 0, 0.5 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(0, 100, 6.2 * 0.3, 9.5 * 0.3, 3/2 * np.pi, 2 * np.pi),
            )
        ]

        
    def iter_zones(self):
        for key, value in const.SHOT_ZONES_BBOX.items():
            # In horizontal layout, x is x and y is y
            x = [
                value["x_lower_bound"],
                value["x_upper_bound"],
                value["x_upper_bound"],
                value["x_lower_bound"],
                value["x_lower_bound"],
            ]
            y = [
                value["y_lower_bound"],
                value["y_lower_bound"],
                value["y_upper_bound"],
                value["y_upper_bound"],
                value["y_lower_bound"],
            ]
            yield key, x, y

class PassVisual(HorizontalPitchVisual):
    def __init__(self, *args, **kwargs):
        self.line_color = 'green'
        self.line_width = 3
        self.shot_color = rgb_to_color(self.bright_blue)
        self.failed_color = rgb_to_color(self.bright_yellow)
        self.bbox_color = rgb_to_color(self.bright_green)
        self.marker_size = 15
        self.basic_text_size = 10
        self.text_size = 20
        super().__init__(*args, **kwargs)

    def add_pass(self,pass_data, pass_id, home_team_color = "green" , away_team_color = "red"):
        
        df_frames = pass_data.df_tracking[pass_data.df_tracking['id'] == pass_id]
        selected_event = pass_data.df_pass[pass_data.df_pass['id'] == pass_id]

        if not df_frames.empty:
            for tn in ['home_team', 'away_team']:
                df_team = df_frames[df_frames['team'] == tn]

                if df_team.empty:
                    continue

                team_direction = df_team.iloc[0]['team_direction']

                # Adjust coordinates only if attacking to the left
                if team_direction == 'left':
                    df_team['x_adjusted'] = 105 - df_team['x_adjusted']
                    df_team['y_adjusted'] = 68 - df_team['y_adjusted']

                is_possession_team = df_team.iloc[0]['team_id'] == selected_event['team_id']
                is_possession_team = is_possession_team.item() if isinstance(is_possession_team, pd.Series) else is_possession_team

                passer_id = selected_event['player_id'].iloc[0] if isinstance(selected_event['player_id'], pd.Series) else selected_event['player_id']
                receiver_id = selected_event['pass_recipient_id'].iloc[0] if isinstance(selected_event['pass_recipient_id'], pd.Series) else selected_event['pass_recipient_id']

                # Locate passer and receiver in team tracking data
                passer = df_team[df_team['player_id'] == passer_id]
                receiver = df_team[df_team['player_id'] == receiver_id]

                # Fallback for passer not in df_team
                if passer.empty:
                    passer = df_frames[df_frames['player_id'] == passer_id]
                    if team_direction == 'left':
                        passer_x = 105 - passer.iloc[0]['x_adjusted']
                        passer_y = 68 - passer.iloc[0]['y_adjusted']
                    else:
                        passer_x = passer.iloc[0]['x_adjusted']
                        passer_y = passer.iloc[0]['y_adjusted']
                else:
                    passer_x = float(passer.iloc[0]['x_adjusted'])
                    passer_y = float(passer.iloc[0]['y_adjusted'])

                # Get and mirror pass end coordinates if needed
                end_x = float(selected_event['end_x'].iloc[0] if isinstance(selected_event['end_x'], pd.Series) else selected_event['end_x'])
                end_y = float(selected_event['end_y'].iloc[0] if isinstance(selected_event['end_y'], pd.Series) else selected_event['end_y'])

                if team_direction == 'left':
                    end_x = 105 - end_x
                    end_y = 68 - end_y

                end_x_norm = end_x * 100 / 105
                end_y_norm = end_y * 100 / 68

                # Plot all players for this team
                player_x = df_team['x_adjusted'] * 100 / 105
                player_y = df_team['y_adjusted'] * 100 / 68

                self.fig.add_trace(
                    go.Scatter(
                        x=player_x,
                        y=player_y,
                        mode="markers",
                        marker=dict(size=10,
                                    color=home_team_color if tn == 'home_team' else away_team_color,
                                    line=dict(width=1, color='black')),
                        name=f"{'Home' if tn == 'home_team' else 'Away'} Players",
                        showlegend=True
                    )
                )

                # Plot pass only for possession team
                if is_possession_team:
                    passer_x_norm = passer_x * 100 / 105
                    passer_y_norm = passer_y * 100 / 68

                    # Passer marker
                    self.fig.add_trace(
                        go.Scatter(
                            x=[passer_x_norm],
                            y=[passer_y_norm],
                            mode="markers",
                            marker=dict(size=12, color="black",
                                        line=dict(width=2, color=home_team_color if tn == 'home_team' else away_team_color)),
                            name="Passer",
                            showlegend=True
                        )
                    )

                    # Receiver and dotted line from pass end
                    if not receiver.empty:
                        receiver_x = float(receiver.iloc[0]['x_adjusted'])
                        receiver_y = float(receiver.iloc[0]['y_adjusted'])

                        receiver_x_norm = receiver_x * 100 / 105
                        receiver_y_norm = receiver_y * 100 / 68

                        self.fig.add_trace(
                            go.Scatter(
                                x=[end_x_norm, receiver_x_norm],
                                y=[end_y_norm, receiver_y_norm],
                                mode="lines",
                                line=dict(dash="dot", color="black", width=1.5),
                                showlegend=False
                            )
                        )

                    # Pass arrow
                    if end_x != -1:
                        self.fig.add_annotation(
                            x=end_x_norm,
                            y=end_y_norm,
                            ax=passer_x_norm,
                            ay=passer_y_norm,
                            xref="x", yref="y", axref="x", ayref="y",
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=2,
                            arrowcolor="black",
                            showarrow=True
                        )
        
        

