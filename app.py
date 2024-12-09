# imports
import streamlit as st
import base64
import io
from io import BytesIO
import requests
import urllib
import numpy as np

from clarifai.client.model import Model
from clarifai.client.app import App
from clarifai.client.auth import create_stub
from clarifai.client.workflow import Workflow
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.modules.css import ClarifaiStreamlitCSS
from annotated_text import annotated_text
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_image_select import image_select

# streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# This must be within the display() function
PAT = "1a7605444bbe4951ab808e8e028bfd33"
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()


promptLLM = """I need your help with image moderation. Please assess the following image for content that falls into any of the following categories: nudity, drugs, violence, gore, and other NSFW (Not Safe For Work) content. Ignore any text overlays, such as 'Image blocked on upload by AI' or existing NSFW scores, as these may be erroneous. For each category, provide a clear evaluation and specify any detected elements that may be of concern.

Assign an NSFW score between 0-100 based on the following categories:

1. Score 0: Safe (0) - The image is entirely safe and appropriate for all audiences. No nudity, violence, drug use, suggestive content, or any other disturbing elements.

2. Score 20: Mildly Suggestive (20) - The image contains mildly suggestive elements that are not explicit but may be slightly provocative or flirtatious.

3. Score 40: Moderately Inappropriate (40) - The image includes moderately inappropriate content, such as revealing clothing, mild violence, or suggestive poses.

4. Score 60: Explicit Content (60) - The image contains explicit content like partial nudity, visible drug paraphernalia, or moderate violence and gore.

5. Score 80: Highly Inappropriate (80) - The image includes highly inappropriate content such as full nudity, significant violence, weapons, guns, knives, or visible drug use.

6. Score 100: Extreme NSFW (100) - The image contains extremely explicit or graphic content, such as pornography, severe violence, or extreme gore.

The goal is to ensure the image is safe for professional, public, and family-friendly environments. Clearly indicate whether the image should be flagged or passed based on the evaluation criteria and the assigned NSFW score, without considering erroneous text. Here is an example output:

"Summary: The image depicts a person with what appears to be blood on their arm and mouth, which could be interpreted as self-harm or violence. Based on the content, here is the evaluation:

Nudity: There is no nudity in the image.

Drugs: There are no visible drugs or drug paraphernalia.

Violence: The image suggests violence or self-harm due to the presence of blood and the way it is depicted.

Weapon: There is no visible weapon, knife, or gun in the image.

Gore: The image contains blood, which can be considered gory.

Other NSFW Content: The image is disturbing due to the implication of self-harm or violence.

NSFW Score: 80 (Highly Inappropriate)

Recommendation: The image should be flagged due to the depiction of blood and the implication of self-harm or violence, making it highly inappropriate for professional, public, and family-friendly environments."
format output as a json-formatted
"""

business_prompt = promptLLM

##########################
#### HELPER FUNCTIONS ####
##########################

def text_color_for_background(hex_code):
    """Determine the appropriate text color (white or black) for a given background color."""
    return "#000000" if is_light_or_dark(hex_code) == "light" else "#ffffff"

def footer(st):
    with open('footer.html', 'r') as file:
        footer = file.read()
        st.write(footer, unsafe_allow_html=True)

def url_picture_to_base64(img_url):
    response = requests.get(img_url)
    return response.content

####################
####  SIDEBAR   ####
####################

with st.sidebar:
    st.caption('Below options are mostly here to help customize any displayed graphics/text.')

    with st.expander('Header Setup'):
        company_logo = st.text_input(label='Banner Url', value='https://upload.wikimedia.org/wikipedia/commons/b/bc/Clarifai_Logo_FC_Web.png')
        company_logo_width = st.slider(label='Banner Width', min_value=1, max_value=1000, value=300)
        page_title = st.text_input(label='Module Title', value='Content Moderation Demo')

    with st.expander('Content Moderation using VLLMs'):
        content_moderation_vllm_subheader_title = st.text_input(label='Anomaly Detection subheader title', value='✨ Leveraging Clarifai for Anomaly Detection ✨')

        st.subheader("Model Selections")

        # Visual Classifier Selection

        community_visual_classifiers = list(
            App(pat=auth._pat).list_models(
                filter_by={"model_type_id": "visual-classifier"}, only_in_app=False
            )
        )
        community_visual_classifiers_ids_only = [x.id for x in community_visual_classifiers]

        print(
            f"community_visual_classifiers_ids_only: {community_visual_classifiers_ids_only}"
        )

        vis_class_model_id = st.selectbox(
            label="Select Image Classification",
            options=community_visual_classifiers_ids_only,
            index=19,  # this should hopefully be the `moderation-all-resnext-2` model selected
        )
        selected_vis_clas_model = [
            x for x in community_visual_classifiers if x.id == vis_class_model_id
        ][0]
        vis_class_model_name = selected_vis_clas_model.name
        vis_class_user_id = selected_vis_clas_model.user_id
        vis_class_app_id = selected_vis_clas_model.app_id

        vis_class_max_concepts = st.slider(
            label="Specify max concepts", min_value=1, max_value=200, value=20
        )

        # LLVM Selection
        st.write("")
        st.write("")

        community_llvms = list(
            App(pat=auth._pat).list_models(
                filter_by={"model_type_id": "multimodal-to-text"}, only_in_app=False
            )
        )
        community_llvms_ids_only = [x.id for x in community_llvms]

        llvm_model_id = st.selectbox(
            label="Select LLVM",
            options=community_llvms_ids_only,
            index=11,  # this should hopefully be the `gpt-4o`
        )

        selected_llvm = [x for x in community_llvms if x.id == llvm_model_id][0]
        llvm_model_name = selected_llvm.name
        llvm_user_id = selected_llvm.user_id
        llvm_app_id = selected_llvm.app_id

        # llvm inference params
        llvm_temp = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0)

        llvm_max_tokens = st.number_input(label="Max Tokens", value=1024)

        llvm_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=0.8)

        technical_prompt = st.text_area(
            label="Technical Prompt",
            value="Only response to the following prompt in a json-formatted list of 30 individual concepts/classes.",
        )
        #### Output Display options
        st.divider()

        st.subheader("Output Display Options")

        tag_bg_color = st.color_picker(label="Tag Background Color (hex)", value="#aabbcc")

        tag_text_color = st.color_picker(label="Tag Text Color (hex)", value="#2B2D37")

    with st.expander('Moderation Recognition'):
        moderation_recognition_subheader_title = st.text_input(label='Insulator Defect Detection Subheader Text', value='✨ Leveraging Clarifai for Insulator Defect Detection ✨')
        inappropriate_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/moderation_1.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/moderation_1.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/moderation_1.png\n'
        )
        box_color = st.color_picker(label='Detection Bounding box Color', value='#0000FF', key='color')
        box_thickness = st.slider(label='Detection Bounding box Thickness', min_value=1, max_value=10, value=3)

        moderation_detection_threshold = st.slider(label='Moderation Detection Threshold', min_value=0.0, max_value=1.0, value=0.3)
        tag_bg_color_1 = st.color_picker(label='Tag Background Color', value='#aabbcc', key='tag_bg_color_1')
        tag_text_color_1 = st.color_picker(label='Tag Text Color', value='#2B2D37', key='tag_text_color_1')

####################
####  MAIN PAGE ####
####################

st.image(company_logo, width=company_logo_width)
st.title(page_title)

tab1, tab2 = st.tabs(['Content Moderation using VLLMs', 'Moderation Recognition'])

##############################
#### Content Moderation using VLLMs ####
##############################

with tab1:
    try:
        st.subheader(content_moderation_vllm_subheader_title)

        sub_tab1, sub_tab2 = st.tabs(["Upload by URL", "Upload by file"])

        # image url version
        with sub_tab1:
            with st.form(key="input-data-url"):

                # url version
                upload_image_url = st.text_input(
                    label="Enter image url:",
                    value="https://samples.clarifai.com/metro-north.jpg",
                )

                business_prompt_url = st.text_area(
                    label="Enter prompt for Large Language Visual Model (LLVM)",
                    value=business_prompt,
                    height=100,
                )

                submitted_url = st.form_submit_button("Upload")

        # local file version
        with sub_tab2:
            with st.form(key="input-data-file"):
                upload_image_file = st.file_uploader(
                    label="Upload image file", type=["jpg", "jpeg", "png", "gif"]
                )

                business_prompt_file = st.text_area(
                    label="Enter prompt for Large Language Visual Model (LLVM)",
                    value=business_prompt,
                    height=100,
                )

                submitted_file = st.form_submit_button("Upload")


        if submitted_url or submitted_file:

            #### display image

            # figure out which one was submitted
            if submitted_url == True:
                st.image(upload_image_url, caption=upload_image_url)
                business_prompt = business_prompt_url
            if submitted_file == True:
                st.image(upload_image_file, caption="Uploaded Image File")
                image_bytes = upload_image_file.read()
                business_prompt = business_prompt_file

            #### Visual Classifier Output

            st.subheader(f"Visual Classifier Output(s):")
            with st.spinner():
                # with st.expander(label=f'{vis_class_model_name}'):
                st.write(f"{vis_class_model_name} | {vis_class_model_id}")

                vis_class_model = Model(
                    pat=auth._pat,
                    model_id=vis_class_model_id,
                    user_id=vis_class_user_id,
                    app_id=vis_class_app_id,
                )

                if submitted_url == True:
                    vis_class_pred = vis_class_model.predict_by_url(
                        url=upload_image_url,
                        input_type="image",
                        inference_params={"max_concepts": vis_class_max_concepts},
                    )

                if submitted_file == True:
                    vis_class_pred = vis_class_model.predict_by_bytes(
                        input_bytes=image_bytes,
                        input_type="image",
                        inference_params={"max_concepts": vis_class_max_concepts},
                    )
                # print(vis_class_pred.outputs[0].data.concepts)

                vis_class_tuple_of_tuples = tuple(
                    [
                        (f"{x.name}", f"{x.value:.3f}", tag_bg_color, tag_text_color)
                        for x in vis_class_pred.outputs[0].data.concepts
                    ]
                )

                # print(f"tuples: {vis_class_tuple_of_tuples}")

                list_with_empty_strings = []
                for item in vis_class_tuple_of_tuples:
                    list_with_empty_strings.append(item)
                    list_with_empty_strings.append(" ")  # Add an empty string after each item

                # Remove the last empty string as it's not needed
                if list_with_empty_strings[-1] == "":
                    list_with_empty_strings.pop()

                # Convert back to a tuple if needed
                vis_class_tuple_of_tuples = tuple(list_with_empty_strings)

                annotated_text(*vis_class_tuple_of_tuples)

            st.subheader(f"Workflow Output(s):")
            with st.spinner():
                workflow_url = "https://clarifai.com/clarifai/Momio-Debug/workflows/Momio-Image-moderation-3"
                # Creating the workflow
                vis_class_workflow = Workflow(
                    url=workflow_url,
                    pat=auth._pat,
                )

                if submitted_url == True:
                    vis_class_pred2 = vis_class_workflow.predict_by_url(
                        url=upload_image_url,
                        input_type="image",
                    )

                if submitted_file == True:
                    vis_class_pred2 = vis_class_workflow.predict_by_bytes(
                        input_bytes=image_bytes,
                        input_type="image",
                    )

                print(vis_class_pred2.results[0].outputs)
                vis_class_tuple_of_tuples2 = ()

                for output2 in vis_class_pred2.results[0].outputs[:]:
                    print(f"model_id: {output2.model.id} **************")
                    print(f"data: *****\n{output2.data}")

                    if (
                        output2.model.id == "moderation-all-resnext-2"
                        or output2.model.id == "nsfw-recognition"
                        or output2.model.id == "moderation-recognition"
                        or output2.model.id == "moderation-multilingual-text-classification"
                        or output2.model.id == "weapon-detection"
                    ):
                        vis_class_tuple_of_tuples2 = vis_class_tuple_of_tuples2 + tuple(
                            [
                                (f"{x.name}", f"{x.value:.3f}", tag_bg_color, tag_text_color)
                                for x in output2.data.concepts
                            ]
                        )

                    if output2.model.id == "weapon-detection" and hasattr(
                        output2.data, "regions"
                    ):
                        weapon_regions = output2.data.regions
                        highest_value = 0.00
                        # loop through the regions and find the highest value of prediction
                        for weapon_region in weapon_regions:
                            print(f"Weapon Data: {weapon_region} ")
                            current_value = float(weapon_region.value)
                            if current_value >= highest_value:
                                highest_value = current_value

                        if highest_value > 0.5:
                            vis_class_tuple_of_tuples2 = vis_class_tuple_of_tuples2 + tuple(
                                [
                                    (
                                        f"Weapon",
                                        f"{highest_value:.3f}",
                                        tag_bg_color,
                                        tag_text_color,
                                    )
                                ]
                            )

                print(f"tuples: {vis_class_tuple_of_tuples2}")

                annotated_text(*vis_class_tuple_of_tuples2)

            ### LLVM Output
            st.subheader(f"LLVM Output(s):")
            with st.spinner():
                llvm_class_model = Model(
                    pat=auth._pat,
                    model_id=llvm_model_id,
                    user_id=llvm_user_id,
                    app_id=llvm_app_id,
                )

                llvm_inference_params = {
                    "temperature": llvm_temp,
                    "max_tokens": llvm_max_tokens,
                    "top_p": llvm_top_p,
                }

                if submitted_url == True:
                    llvm_pred = llvm_class_model.predict(
                        inputs=[
                            Inputs.get_multimodal_input(
                                input_id="",
                                image_url=upload_image_url,
                                raw_text=f"{business_prompt}",
                            )
                        ],
                        inference_params=llvm_inference_params,
                    )

                if submitted_file == True:
                    llvm_pred = llvm_class_model.predict(
                        inputs=[
                            Inputs.get_multimodal_input(
                                input_id="",
                                image_bytes=image_bytes,
                                raw_text=f"{business_prompt}",
                            )
                        ],
                        inference_params=llvm_inference_params,
                    )

                # wrangling output and cleaning up / converting to labels
                llvm_output = llvm_pred.outputs[0].data.text.raw

                print(f"llvm_output:\n{llvm_output}")
                st.write(llvm_output)
        
        
        # Collapsible text field at the bottom
        with st.expander("Details"):
            st.markdown("""
            ### About Anomaly Detection
            Anomaly detection automatically identifies unusual patterns or defects in tablet pills that deviate from expected normal conditions. This helps in quality control by detecting manufacturing defects like chips, cracks, or contamination.
            
            ### How This Works
            1. **Image Selection**: 
                - Choose one of the tablet pill images from the Carousel
                - Each image shows different types of defects (chips, breaks, or dirt)
            
            2. **Output**:
                - Click "Run Anomaly Detection" to analyze the selected image
                - You'll see three views:
                    * Original Image: The unprocessed tablet photo
                    * Heat Map: Shows detected anomalies where:
                        - Darker red areas = Higher likelihood of defects
                        - Lighter/Black areas = Normal conditions
                    * Composite: Combines original image with heatmap for easy defect location
            
            ### Original Model
            This implementation uses Clarifai's Anomaly Detection model for tablet pills. 
            - View the App here: [Clarifai Pill Anomaly Detection](https://clarifai.com/clarifai/anomaly-detection-tablet-pills)
            """)
            
            # Original text area for additional notes
            project_details = st.text_area(
                "Add Your Notes:",
                height=100,
                placeholder="Add any additional notes..."
            )
            if project_details:
                st.markdown("### Your Notes:")
                st.write(project_details)
                            
    except Exception as e:
      st.error(f"Error in VLLMs tab: {str(e)}")

# #########################
# #### Moderation Recognition ####
# #########################

# with tab2:
#     try:
#         st.subheader(moderation_recognition_subheader_title)

#         threshold = moderation_detection_threshold
        
#         img = image_select(
#             label="Select image:",
#             images=inappropriate_images.split('\n'),
#             captions=["Image #1", "Image #2", "Image #3"]
#         )

#         if st.button("Run Moderation Detection"):
#             st.divider()
            
#             model_url = "https://clarifai.com/clarifai/insulator-defect-detection/models/insulator-condition-inception"
            
#             with st.spinner("Processing moderation detection..."):
#                 model = Model(url=model_url, pat=PAT)
#                 model_prediction = model.predict_by_url(image_url, input_type="image")
#                 concepts = model_prediction.outputs[0].data.concepts

#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.write('Original')
#                     im1_pil = Image.open(urllib.request.urlopen(img))
#                     st.image(im1_pil)

#                 with col2:
#                     st.write("Moderation Detection Output")
#                     filtered_concepts = [
#                         x for x in surface_class_pred.outputs[0].data.concepts 
#                         if x.value >= threshold
#                     ]
                    
#                     if not filtered_concepts:
#                         st.info(f"No concepts detected above the confidence threshold of {threshold:.2f}")
#                     else:
#                         concept_data = tuple([
#                             (f'{x.name}', f'{x.value:.3f}', tag_bg_color_2, tag_text_color_2)
#                             for x in filtered_concepts
#                         ])
                        
#                         # Add spacing between items
#                         list_with_empty_strings = []
#                         for item in concept_data:
#                             list_with_empty_strings.append(item)
#                             list_with_empty_strings.append(" ")
                        
#                         if list_with_empty_strings and list_with_empty_strings[-1] == " ":
#                             list_with_empty_strings.pop()
                        
#                         concept_data = tuple(list_with_empty_strings)
#                         annotated_text(*concept_data)

#         with st.expander("Details"):
#           st.markdown("""
#                       ### About Moderation Detection
#                       Recognizes inappropriate content in images and video containing concepts: gore, drug, explicit, suggestive, and safe.
                      
#                       ### How This Works
#                       1. **Image Selection**: 
#                           - Choose one of the images from the carousel
#                           - Each image shows different types of potential defects
                      
#                       2. **Output**:
#                           - Click "Run Moderation Detection" to analyze the selected image
#                           - You'll see two views:
#                               * Original Image: The unprocessed image
#                               * Predicted Defects: Shows bounding boxes around detected defects with:
#                                   - Labels indicating the type of defect detected either "Broken Part" or "Flash Over"
#                                   - Confidence scores for each detection
                      
#                       ### Original Model
#                       This implementation uses Clarifai's Moderation Detection model.
#                       - View the App here: [Clarifai Moderation Defect Model](https://clarifai.com/clarifai/main/models/moderation-recognition)
#                       """)
#           project_details = st.text_area(
#                                         "Add Your Notes:",
#                                         height=100,
#                                         key="moderation_detection_notes",
#                                         placeholder="Add any additional notes about moderation detection..."
#                                         )
#           if project_details:
#                 st.markdown("### Your Notes:")
#                 st.write(project_details)
#     except Exception as e:
#         st.error(f"Error in Defect Detection tab: {str(e)}")


####################
####  FOOTER    ####
####################

footer(st)