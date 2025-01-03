import pymongo
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate
from reportlab.lib.colors import Color, yellow, green, red, HexColor
from io import BytesIO
from params import PROD_MONGODB_CONN_STRING, PROD_DB_NAME, PROD_INITIAL_BENCHMARK_COLLECTION

# Connect to MongoDB and retrieve data
client = pymongo.MongoClient(PROD_MONGODB_CONN_STRING)
db = client[PROD_DB_NAME]
collection = db[PROD_INITIAL_BENCHMARK_COLLECTION]
data = list(collection.find({}, {'_id': 0, 'context_precision': 1, 'context_recall': 1, 'faithfulness': 1, 'context_relevancy': 1, 'answer_relevancy': 1, 'response_time': 1}))
df = pd.DataFrame(data)

# Clean the data
df['response_time'] = df['response_time'].replace([np.inf, -np.inf], np.nan).dropna()

# Calculate median scores
median_scores = {
    'Median Context Precision': df['context_precision'].median(),
    'Median Context Recall': df['context_recall'].median(),
    'Median Faithfulness': df['faithfulness'].median(),
    'Median Context Relevancy': df['context_relevancy'].median(),
    'Median Answer Relevancy': df['answer_relevancy'].median()
}

# Radar chart function
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return {
                'polar': Spine.circular_spine(self, (0.5, 0.5), 0.5)
            }

    register_projection(RadarAxes)
    return theta

# Create radar chart
def create_radar_chart(median_scores):
    categories = list(median_scores.keys())
    N = len(categories)
    theta = radar_factory(N, frame='polygon')
    values = list(median_scores.values())

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='radar'))
    ax.plot(theta, values)
    ax.fill(theta, values, alpha=0.25)
    ax.set_varlabels(categories)

    ax.set_ylim(0, 1)
    plt.title("RAG Pipeline Performance Benchmark (Median)", pad=20, fontsize=12)

    return fig

# Create radar chart
radar_chart = create_radar_chart(median_scores)

# Save radar chart to BytesIO object
radar_img_buffer = BytesIO()
radar_chart.savefig(radar_img_buffer, format='png', dpi=300, bbox_inches='tight')
radar_img_buffer.seek(0)

# Create histograms
def create_histogram(data, metric_name):
    plt.figure(figsize=(5, 3))
    
    # Create histogram
    counts, bins, _ = plt.hist(data, bins=10, range=(0, 1), edgecolor='black')
    
    plt.title(f"{metric_name}", fontsize=10)
    plt.xlabel("Metric Value")
    plt.ylabel("Number of Data Points")
    plt.xlim(0, 1)
    
    # Adjust y-axis
    max_count = max(counts)
    plt.ylim(0, max_count * 1.1)  # Set y-axis limit to 110% of max count
    
    # Determine appropriate y-axis tick interval
    if max_count > 50:
        y_interval = 10
    elif max_count > 20:
        y_interval = 5
    else:
        y_interval = 2
    
    plt.yticks(range(0, int(max_count * 1.1) + 1, y_interval))
    
    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(bins[i] + (bins[i+1] - bins[i])/2, count, f'{int(count)}', 
                 ha='center', va='bottom')
    
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    # Adjust layout
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

histogram_buffers = {
    'Context Precision': create_histogram(df['context_precision'], 'Context Precision'),
    'Context Recall': create_histogram(df['context_recall'], 'Context Recall'),
    'Faithfulness': create_histogram(df['faithfulness'], 'Faithfulness'),
    'Context Relevancy': create_histogram(df['context_relevancy'], 'Context Relevancy'),
    'Answer Relevancy': create_histogram(df['answer_relevancy'], 'Answer Relevancy')
}

# Create execution time histogram
def create_execution_time_histogram(data):
    plt.figure(figsize=(7, 4))  # Increased width for better readability
    
    # Define bins
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
    
    # Create histogram
    hist, bin_edges = np.histogram(data['response_time'], bins=bins)
    
    # Plot bars
    plt.bar(range(len(hist)), hist, align='center', edgecolor='black')
    
    # Set x-axis ticks and labels
    plt.xticks(range(len(hist)), ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+'])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set labels and title
    plt.title("Execution Time Distribution", fontsize=12)
    plt.xlabel("Seconds", fontsize=10)
    plt.ylabel("Number of Requests", fontsize=10)
    
    # Adjust y-axis
    max_count = max(hist)
    plt.ylim(0, max_count * 1.1)  # Set y-axis limit to 110% of max count
    
    # Determine appropriate y-axis tick interval
    if max_count > 50:
        y_interval = 10
    elif max_count > 20:
        y_interval = 5
    else:
        y_interval = 2
    
    plt.yticks(range(0, int(max_count * 1.1) + 1, y_interval))
    
    # Add value labels on top of each bar
    for i, v in enumerate(hist):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def categorize_result(context_recall, context_precision, faithfulness, answer_relevancy):
    retrieval_threshold = 0.7
    answer_threshold = 0.7

    retrieval_correct = (context_recall + context_precision) / 2 > retrieval_threshold
    answer_correct = (faithfulness + answer_relevancy) / 2 > answer_threshold

    return 'correct' if retrieval_correct else 'incorrect', 'correct' if answer_correct else 'incorrect'

def create_confusion_matrix(df):
    confusion_matrix = {
        ('incorrect', 'incorrect'): 0,
        ('incorrect', 'correct'): 0,
        ('correct', 'incorrect'): 0,
        ('correct', 'correct'): 0
    }

    for _, row in df.iterrows():
        retrieval, answer = categorize_result(
            row['context_recall'],
            row['context_precision'],
            row['faithfulness'],
            row['answer_relevancy']
        )
        confusion_matrix[(retrieval, answer)] += 1

    total = sum(confusion_matrix.values())
    confusion_percentages = {k: (v / total) * 100 for k, v in confusion_matrix.items()}

    return confusion_percentages

def create_confusion_matrix_plot(confusion_percentages):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    retrieval = ['Incorrect', 'Correct']
    answer = ['Incorrect', 'Correct']
    
    cell_colors = [
        ['#FF6B6B', '#FF6B6B'],  # Yellow, Green
        ['#FFA500', '#4ECDC4']   # Red, Red
    ]
    
    # Create the main grid
    for i, r in enumerate(retrieval):
        for j, a in enumerate(answer):
            value = confusion_percentages[(r.lower(), a.lower())]
            color = cell_colors[i][j]
            rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            text_color = 'black' if color == '#FFFF00' else 'white'
            ax.text(j + 0.5, i + 0.5, f'{value:.1f}%', ha='center', va='center', color=text_color, fontweight='bold', fontsize=14)

    # Set limits and remove axes
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 2.2)
    ax.axis('off')
    
    # Add labels
    ax.text(1, 2.15, 'Retrieval', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(-0.15, 1, 'Answer', ha='right', va='center', fontsize=14, fontweight='bold', rotation=90)
    
    # Add x-axis labels for Retrieval on top
    for i, label in enumerate(['Incorrect', 'Correct']):
        ax.text(i + 0.5, 2.05, label, ha='center', va='bottom', fontsize=12)

    # Add y-axis labels for Answer
    for i, label in enumerate(['Incorrect', 'Correct']):
        ax.text(-0.05, i + 0.5, label, ha='right', va='center', fontsize=12)

    # Add annotations and arrows
    arrow_props = dict(arrowstyle='->', color='black', linewidth=1.5)
    
    # Bottom left
    ax.annotate('Incorrect retrieval - tune your search', xy=(0.1, -0.1), xytext=(-0.1, -0.2), 
                ha='left', va='center', fontsize=10, annotation_clip=False, arrowprops=arrow_props)
    
    # Top left
    ax.annotate('Incomplete retrieval but answer was\npartially correct - tune chunking', xy=(0.1, 2.1), xytext=(-0.1, 2.2), 
                ha='left', va='center', fontsize=10, annotation_clip=False, arrowprops=arrow_props)
    
    # Top right
    ax.annotate('Correct answer on correct retrieval', xy=(1.9, 2.1), xytext=(2.1, 2.2), 
                ha='right', va='center', fontsize=10, annotation_clip=False, arrowprops=arrow_props)
    
    # Bottom right
    ax.annotate('Correct retrieval but wrong answer >\nTune the prompt / model params', xy=(1.9, -0.1), xytext=(2.1, -0.2), 
                ha='right', va='center', fontsize=10, annotation_clip=False, arrowprops=arrow_props)

    plt.tight_layout()
    return fig

def analyze_results(confusion_percentages, df):
    analysis = []
    if confusion_percentages[("incorrect", "incorrect")] > 20:
        analysis.append(f"High rate of incorrect retrieval and answers ({confusion_percentages[('incorrect', 'incorrect')]:.1f}%). Consider tuning your search algorithm. Median Context Recall: {df['context_recall'].median():.2f}, Median Context Precision: {df['context_precision'].median():.2f}")
    
    if confusion_percentages[("incorrect", "correct")] > 10:
        analysis.append(f"Significant rate of correct answers despite incorrect retrieval ({confusion_percentages[('incorrect', 'correct')]:.1f}%). This suggests the model might be using prior knowledge. Median Faithfulness score: {df['faithfulness'].median():.2f}")
    
    if confusion_percentages[("correct", "incorrect")] > 10:
        analysis.append(f"High rate of incorrect answers despite correct retrieval ({confusion_percentages[('correct', 'incorrect')]:.1f}%). Consider tuning your prompts or model parameters. Median Answer Relevancy: {df['answer_relevancy'].median():.2f}")
    
    return analysis

# Prepare data for the report
descriptions = {
    'Median Context Precision': 'Measures how much of the retrieved context is actually relevant.',
    'Median Context Recall': 'Measures how much of the relevant information is retrieved.',
    'Median Faithfulness': 'Measures how well the answer aligns with the provided context.',
    'Median Context Relevancy': 'Measures how relevant the retrieved context is to the question.',
    'Median Answer Relevancy': 'Measures how relevant the generated answer is to the question.'
}

ranges = {
    'Median Context Precision': 'Aim for > 0.65',
    'Median Context Recall': 'Aim for > 0.75',
    'Median Faithfulness': 'Aim for > 0.85',
    'Median Context Relevancy': 'Aim for > 0.75',
    'Median Answer Relevancy': 'Aim for > 0.85'
}

improvements = {
    'Median Context Precision': [
        '    • Refine document chunking strategy',
        '    • Improve relevance scoring in retrieval'
    ],
    'Median Context Recall': [
        '    • Expand knowledge base with diverse sources',
        '    • Enhance semantic search capabilities'
    ],
    'Median Faithfulness': [
        '    • Fine-tune language model on domain-specific data',
        '    • Improve prompt engineering for accuracy'
    ],
    'Median Context Relevancy': [
        '    • Enhance context ranking algorithms',
        '    • Implement context re-ranking post-retrieval'
    ],
    'Median Answer Relevancy': [
        '    • Refine answer generation prompts',
        '    • Implement answer quality checks'
    ]
}

# Create PDF
pdf_buffer = BytesIO()
doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=108, bottomMargin=18)

# Styles
styles = getSampleStyleSheet()
title_style = styles['Title']
heading_style = styles['Heading2']
body_style = styles['BodyText']

metric_style = ParagraphStyle(
    'MetricStyle',
    parent=styles['Heading3'],
    spaceAfter=6,
    textColor=Color(0.2, 0.2, 0.6)
)

description_style = ParagraphStyle(
    'DescriptionStyle',
    parent=body_style,
    spaceAfter=6,
    textColor=Color(0.3, 0.3, 0.3)
)

range_style = ParagraphStyle(
    'RangeStyle',
    parent=body_style,
    spaceAfter=6,
)

improvement_style = ParagraphStyle(
    'ImprovementStyle',
    parent=body_style,
    spaceAfter=6,
    textColor=HexColor('#CCCC00') 
)

bullet_style = ParagraphStyle(
    'BulletStyle',
    parent=body_style,
    leftIndent=20,
    spaceAfter=3
)

# Function to determine if a metric is above its target range
def is_above_target(metric, value):
    target = float(ranges[metric].split('>')[1].strip())
    return value > target

# Header function
def header(canvas, doc):
    canvas.saveState()
    
    logo = Image('YourCompanyLogo.png', width=2.5*inch, height=1.5*inch)
    logo.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - 1.5*inch)
    
    # Place creator text on the same line as logo, aligned to top right
    creator = Paragraph("Created by Daniel D'souza", styles['Normal'])
    w, h = creator.wrap(doc.width / 2, doc.topMargin)  # Limit width to half the page
    creator.drawOn(canvas, doc.width + doc.leftMargin - w - 72, doc.height + doc.topMargin - 14)
    
    canvas.restoreState()

# Add page template
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
template = PageTemplate(id='test', frames=frame, onPage=header)
doc.addPageTemplates([template])

# Content
content = []

# Title
content.append(Paragraph("RAG Pipeline Performance Benchmark", title_style))
content.append(Spacer(1, 24))  # Increased spacing after title

# Introduction
content.append(Paragraph("This report provides an initial benchmark of our RAG (Retrieval-Augmented Generation) pipeline performance. The radar chart below visualizes the median scores for key metrics, followed by detailed explanations, improvement suggestions, and distribution histograms.", body_style))
content.append(Spacer(1, 12))

# Add radar chart
content.append(Image(radar_img_buffer, width=5*inch, height=5*inch))
content.append(Spacer(1, 12))

# Add a page break before Metric Details
content.append(PageBreak())

# Metrics details
content.append(Paragraph("Metric Details", heading_style))
content.append(Spacer(1, 3))

for metric, score in median_scores.items():
    content.append(Paragraph(f"{metric}: {score:.2f}", metric_style))

    # Determine color based on whether the metric is above target
    color = green if is_above_target(metric, score) else red
    range_style_dynamic = ParagraphStyle(
        'RangeStyleDynamic',
        parent=range_style,
        textColor=color
    )

    content.append(Paragraph(f"Description: {descriptions[metric]}", description_style))
    content.append(Paragraph(f"Target Range: {ranges[metric]}", range_style_dynamic))
    content.append(Paragraph("To Improve:", improvement_style))
    
    for improvement in improvements[metric]:
        content.append(Paragraph(improvement, bullet_style))
    content.append(Spacer(1, 3))

# Histogram section
content.append(Paragraph("Metric Distributions", heading_style))
content.append(Spacer(1, 6))

for metric in median_scores.keys():
    histogram_metric = metric.replace('Median ', '')
    content.append(Image(histogram_buffers[histogram_metric], width=5*inch, height=3*inch))
    content.append(Spacer(1, 12))

content.append(PageBreak())

# Create confusion matrix
confusion_percentages = create_confusion_matrix(df)

# Create confusion matrix plot
confusion_matrix_fig = create_confusion_matrix_plot(confusion_percentages)
confusion_matrix_buffer = BytesIO()
confusion_matrix_fig.savefig(confusion_matrix_buffer, format='png', dpi=300, bbox_inches='tight')
confusion_matrix_buffer.seek(0)
plt.close(confusion_matrix_fig)

# Confusion Matrix section
content.append(Paragraph("Confusion Matrix", heading_style))
content.append(Spacer(1, 6))
content.append(Image(confusion_matrix_buffer, width=5.5*inch, height=5.5*inch))  
content.append(Spacer(1, 6))
content.append(Paragraph("The confusion matrix above provides a visual representation of our RAG pipeline's performance in terms of retrieval and answer generation. It helps us identify areas where the system excels and where it needs improvement.", body_style))
content.append(Spacer(1, 6))


# Add analysis
content.append(Paragraph("Confusion Matrix Analysis", heading_style))
content.append(Spacer(1, 3))
analysis = analyze_results(confusion_percentages, df)
for point in analysis:
    content.append(Paragraph(point, body_style))
    content.append(Spacer(1, 3))
content.append(Paragraph(f"Overall, the confusion matrix shows that our RAG pipeline achieves correct retrieval and answers {confusion_percentages[('correct', 'correct')]:.1f}% of the time. This baseline performance provides a foundation for future improvements in both retrieval and answer generation processes.", body_style))
content.append(Spacer(1, 6))

# Add a page break before Execution Time section
content.append(PageBreak())

# Execution Time section
content.append(Paragraph("Execution Time Distribution", heading_style))
content.append(Spacer(1, 6))

execution_time_histogram = create_execution_time_histogram(df)
content.append(Image(execution_time_histogram, width=5*inch, height=3*inch))
content.append(Spacer(1, 12))

# Explanatory text
median_execution_time = df['response_time'].median()
content.append(Paragraph(f"The median execution time is {median_execution_time:.2f} seconds.", body_style))
content.append(Spacer(1, 6))
content.append(Paragraph("This histogram shows the distribution of execution times for the RAG pipeline initial benchmark. Each bar represents the number of requests that were completed within the corresponding time range.", body_style))

# Build PDF
doc.build(content)
pdf_buffer.seek(0)

# Save PDF
with open("RAG_Pipeline_Benchmark.pdf", "wb") as f:
    f.write(pdf_buffer.getvalue())

print("PDF report generated: RAG_Pipeline_Benchmark.pdf")