import streamlit as st
import pandas as pd
import re
from fuzzywuzzy import fuzz
import base64
from datetime import datetime
import string


def clean_zipcode(zipcode):
    zipcode = str(zipcode)
    if zipcode.startswith("'"):
        zipcode = zipcode[1:]
    if '-' in zipcode:
        zipcode = zipcode.split('-')[0]
    return zipcode


def parse_dates(date_series, date_format):
    # Try to convert and catch parsing errors
    try:
        return pd.to_datetime(date_series, format=date_format, errors='coerce').dt.date
    except Exception as e:
        st.error(f"Date parsing error: {e}")
        return date_series
    

def remove_punctuation(text):
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method to remove punctuation
    return text.translate(translator)



def apply_substitutions(address):
    substitutions = {
        'street': 'St',
        'avenue': 'Ave',
        'boulevard': 'Blvd',
        'road': 'Rd',
        'lane': 'Ln',
        'drive': 'Dr',
        'court': 'Ct',
        'parkway': 'Pkwy',
        'place': 'Pl',
        'square': 'Sq',
        'terrace': 'Terr',
        'trail': 'Trl',
        'apartment': 'Apt',
        'floor': 'Fl',
        'suite': 'Ste',
        'north': 'N',
        'south': 'S',
        'east': 'E',
        'west': 'W',
        'northeast': 'NE',
        'northwest': 'NW',
        'southeast': 'SE',
        'southwest': 'SW',
        'hollow': 'holw',
        'circle': 'cir',
        'curve': 'curv'
    }

    if isinstance(address, float) and pd.isna(address):
        return address
    address = str(address)
    for old, new in substitutions.items():
        address = re.sub(old, new, address)
    return address

def preprocess_zipcode(text):
    if isinstance(text, str):
        # Remove leading zeros
        text = text.lstrip('0')
        # Replace numeric digits with corresponding letters
        text = text.replace('1', 'a').replace('2', 'b').replace('3', 'c').replace('4', 'd')\
                   .replace('5', 'e').replace('6', 'f').replace('7', 'g').replace('8', 'h')\
                   .replace('9', 'i').replace('0', 'j')
        return text.lower().strip()
    return ''

def preprocess_text(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ''

def split_address(address):
    po_box_match = re.match(r'^(po box \d+)\s*(.*)', address, re.IGNORECASE)
    if po_box_match:
        return po_box_match.group(1), po_box_match.group(2)
    
    street_match = re.match(r'^(\d+)\s+(.+)', address)
    if street_match:
        return street_match.group(1), street_match.group(2)
    
    # If no match, return None for street number and the full address for street name
    return None, address




# Define a wrapper function to handle errors
def safe_split_address(address, index):
    try:
        return pd.Series(split_address(address))
    except Exception as e:
        print(f"Error at index {index}: {e}")
        return pd.Series([None, address]) 

def match_columns(row, catalog_row, name_threshold=10, address_threshold=60):
    name_match = fuzz.token_sort_ratio(row['billing_name'], catalog_row['name'])
    
    # Exact match for billing street number
    if catalog_row['street_number'] and row['billing_street_number'] and catalog_row['street_number'] == row['billing_street_number']:
        billing_street_number_match = 100
    else:
        billing_street_number_match = 0
    
    # Fuzzy match score for billing street name
    if catalog_row['street_name'] and row['billing_street_name']:
        billing_address_match = fuzz.token_sort_ratio(row['billing_street_name'], catalog_row['street_name'])
    else:
        billing_address_match = 0
    
    # Exact match for shipping street number
    if catalog_row['street_number'] == row['shipping_street_number'] and catalog_row['street_number'] == row['shipping_street_number']:
        shipping_street_number_match = 100
    else:
        shipping_street_number_match = 0
    
    # Fuzzy match score for shipping street name
    if catalog_row['street_name'] and row['shipping_street_name']:
        shipping_address_match = fuzz.token_sort_ratio(row['shipping_street_name'], catalog_row['street_name'])
    else:
        shipping_address_match = 0
    
    # Determine match results based on thresholds
    name_match_result = name_match if name_match >= name_threshold else 0
    billing_address_match_result = billing_address_match if billing_address_match >= address_threshold else 0
    shipping_address_match_result = shipping_address_match if shipping_address_match >= address_threshold else 0
    
    return name_match_result, billing_address_match_result, shipping_address_match_result, billing_street_number_match, shipping_street_number_match


column_data = {
    
    "Column Name": ["address", "mp_add2", "city", "state", "name"],
    "Description": [
        "Their main address",
        "Optional! Secondary address. Some catalogs have a different column for apartment number or unit for example",
        "City name",
        "State name",
        "Customer full name"
    ],
}
column_df = pd.DataFrame(column_data)

def main():
    st.title("Heaven 2.0")
    st.subheader("Catalog Column Names")
    st.write("Below are the column names the code expects to see. The secondary address column (mp_add2) is there in some catalogs but not others, hence the code will work with or without it. Capitalisation doesn't matter! Any other column you want can be included and won't cause any issues.")
    st.table(column_df)
    # Upload orders CSV file
    orders_file = st.file_uploader("Upload your Orders CSV file", type="csv")

    # Upload catalog Excel file
    catalog_file = st.file_uploader("Upload your Catalog Excel file", type=["xlsx", "xls"])

    if orders_file is not None and catalog_file is not None:
        if st.button("Clean Orders"):
            try:
                orders_df = pd.read_csv(orders_file, encoding='utf-8')
            except UnicodeDecodeError:
                orders_df = pd.read_csv(orders_file, encoding='latin1')

            # Initialize progress bar for orders cleaning
            progress_bar = st.progress(0)
            num_rows = len(orders_df)
            batch_size = num_rows // 5
            cleaned_chunks = []

            # Apply cleaning operations and update progress
            for i in range(0, num_rows, batch_size):
                end_idx = min(i + batch_size, num_rows)
                orders_chunk = orders_df.iloc[i:end_idx].copy()

                # Clean orders_chunk
                orders_chunk = orders_chunk.dropna(subset=['Total'])
                orders_chunk['Billing Zip'] = orders_chunk['Billing Zip'].apply(clean_zipcode)
                orders_chunk['Shipping Zip'] = orders_chunk['Shipping Zip'].apply(clean_zipcode)
                orders_chunk.rename(columns={
                    'Billing Name': 'billing_name',
                    'Billing Street': 'billing_address1',
                    'Billing Zip': 'billing_zip',
                    'Shipping Name': 'shipping_name',
                    'Shipping Street': 'shipping_address1',
                    'Shipping Zip': 'shipping_zip',
                    'Discount Code': 'discount_code'
                }, inplace=True)
                
                orders_chunk['billing_name'] = orders_chunk['billing_name'].apply(preprocess_text)
                orders_chunk['billing_address1'] = orders_chunk['billing_address1'].apply(preprocess_text)
                orders_chunk['shipping_name'] = orders_chunk['shipping_name'].apply(preprocess_text)
                orders_chunk['shipping_address1'] = orders_chunk['shipping_address1'].apply(preprocess_text)
                orders_chunk['billing_name'] = orders_chunk['billing_name'].apply(remove_punctuation)
                orders_chunk['billing_address1'] = orders_chunk['billing_address1'].apply(remove_punctuation)
                orders_chunk['shipping_name'] = orders_chunk['shipping_name'].apply(remove_punctuation)
                orders_chunk['shipping_address1'] = orders_chunk['shipping_address1'].apply(remove_punctuation)
                orders_chunk['billing_address1'] = orders_chunk['billing_address1'].apply(apply_substitutions)
                orders_chunk['shipping_address1'] = orders_chunk['shipping_address1'].apply(apply_substitutions)
                orders_chunk['billing_address1'] = orders_chunk['billing_address1'].apply(preprocess_text)
                orders_chunk['shipping_address1'] = orders_chunk['shipping_address1'].apply(preprocess_text)
                orders_chunk['billing_zip'] = orders_chunk['billing_zip'].astype(str).apply(preprocess_zipcode)
                orders_chunk['shipping_zip'] = orders_chunk['shipping_zip'].astype(str).apply(preprocess_zipcode)


                billing_street_numbers = []
                billing_street_names = []
                shipping_street_numbers = []
                shipping_street_names = []

                # Iterate through each row to apply the safe_split_address function for billing addresses
                for idx, address in orders_chunk['billing_address1'].items():
                    billing_street_number, billing_street_name = safe_split_address(address, idx)
                    billing_street_numbers.append(billing_street_number)
                    billing_street_names.append(billing_street_name)

                # Iterate through each row to apply the safe_split_address function for shipping addresses
                for idx, address in orders_chunk['shipping_address1'].items():
                    shipping_street_number, shipping_street_name = safe_split_address(address, idx)
                    shipping_street_numbers.append(shipping_street_number)
                    shipping_street_names.append(shipping_street_name)

                # Assign the results back to the DataFrame
                orders_chunk['billing_street_number'] = billing_street_numbers
                orders_chunk['billing_street_name'] = billing_street_names
                orders_chunk['shipping_street_number'] = shipping_street_numbers
                orders_chunk['shipping_street_name'] = shipping_street_names


                # Append the cleaned chunk to the list
                cleaned_chunks.append(orders_chunk)

                # Update progress bar
                progress_bar.progress(end_idx / num_rows)

            # Combine all cleaned chunks into the final DataFrame
            orders_df = pd.concat(cleaned_chunks, ignore_index=True)
            columns_to_keep = [
                'Name', 'billing_name', 'billing_address1', 'billing_zip',
                'shipping_name', 'shipping_address1', 'shipping_zip',
                'discount_code', 'billing_street_number', 'billing_street_name',
                'shipping_street_number', 'shipping_street_name', 'Total'
            ]
            
            # After cleaning, filter the DataFrame to include only these columns
            orders_df = orders_df[columns_to_keep]
            st.session_state['orders_df'] = orders_df
            st.success("Orders file has been cleaned successfully!")
            st.write(f"Number orders: {len(orders_df)}")
            if len(orders_df) == 0:
                st.warning("No orders found.")
            

        
        if st.button("Clean Catalog"):
            try:
                catalog_df = pd.read_excel(catalog_file, engine='openpyxl')
            except ValueError as e:
                st.error(f"Error reading the Excel file: {e}")
                return

            # Clean catalog_df
            catalog_df.columns = catalog_df.columns.str.lower()
            required_columns = {"name", "address", "zip"}

            # Check if all required columns are present in catalog_df
            missing_columns = required_columns - set(catalog_df.columns)
           
            if missing_columns:
                st.error(f"The following required columns are missing from the catalog file: {', '.join(missing_columns)}")
            else:
                st.write("")



            

            # Initialize progress bar
            progress_bar = st.progress(0)
            num_chunks = 100  # Number of chunks
            chunk_size = len(catalog_df) // num_chunks

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(catalog_df)

                catalog_chunk = catalog_df.iloc[start_idx:end_idx]

                # Preprocess columns
                
                catalog_chunk['zip'] = catalog_chunk['zip'].apply(clean_zipcode)
                catalog_chunk['zip'] = catalog_chunk['zip'].astype(str).apply(preprocess_zipcode)
                catalog_chunk['name'] = catalog_chunk['name'].apply(preprocess_text)
                catalog_chunk['address'] = catalog_chunk['address'].apply(preprocess_text)
                catalog_chunk['address'] = catalog_chunk['address'].apply(remove_punctuation)
                catalog_chunk['address'] = catalog_chunk['address'].apply(apply_substitutions)
                catalog_chunk['address'] = catalog_chunk['address'].apply(preprocess_text)

                # Update catalog_df with cleaned chunk
                catalog_df.iloc[start_idx:end_idx] = catalog_chunk

                # Update progress bar
                progress_bar.progress((i + 1) / num_chunks)

            catalog_df[['street_number', 'street_name']] = catalog_df['address'].apply(lambda x: pd.Series(split_address(x)))    
            st.session_state['catalog_df'] = catalog_df
            st.success("Catalog file has been cleaned successfully!")
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
         







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        if st.button("Run Matching"):
            if 'orders_df' not in st.session_state:
                st.error("Please clean the orders and catalog files first.")
            elif 'catalog_df' not in st.session_state:
                st.error("Please clean the catalog file first.")
            else:
                orders_df = st.session_state['orders_df']
                catalog_df = st.session_state['catalog_df']

                # Initialize match columns and columns to store TT1 and TT2 in catalog_df
                
                matches = []
                num_rows = len(catalog_df)
                
                progress_placeholder = st.empty()

                
                zip_codes_in_orders = orders_df['billing_zip'].unique().tolist() + orders_df['shipping_zip'].unique().tolist()
                zip_codes_in_orders = list(set(zip_codes_in_orders))  # Remove duplicates
                catalog_df = catalog_df[catalog_df['zip'].isin(zip_codes_in_orders)]
                
                update_interval = 1
                for index, catalog_row in catalog_df.iterrows():
                    match_count = 0
                    removed_rows = pd.DataFrame()
                
                    while match_count < 5:
                        filtered_orders = orders_df[
                            ((orders_df['billing_zip'] == catalog_row['zip']) | (orders_df['shipping_zip'] == catalog_row['zip']))
                        ]
                        
                        if filtered_orders.empty:
                            break

                        best_name_match = best_billing_address_match = best_shipping_address_match = 0
                        best_billing_street_number_match = best_shipping_street_number_match = 0
                        best_row_data = None
                        
                        for _, row in filtered_orders.iterrows():
                            name_match, billing_address_match, shipping_address_match, billing_street_number_match, shipping_street_number_match = match_columns(row, catalog_row)

                            if (name_match > best_name_match or 
                                billing_address_match > best_billing_address_match or 
                                shipping_address_match > best_shipping_address_match or 
                                billing_street_number_match == 100 or 
                                shipping_street_number_match == 100):
                                
                                best_name_match = name_match
                                best_billing_address_match = billing_address_match
                                best_shipping_address_match = shipping_address_match
                                best_billing_street_number_match = billing_street_number_match
                                best_shipping_street_number_match = shipping_street_number_match
                                best_row_data = row

                        if best_row_data is not None:
                            # Store the results directly in the catalog dataframe (this eliminates repeated `loc` calls)
                            matches.append({
                                'name_match': best_name_match,
                                'billing_address_match': best_billing_address_match,
                                'shipping_address_match': best_shipping_address_match,
                                'street_number_match': best_billing_street_number_match,
                                'shipping_street_number_match': best_shipping_street_number_match,
                                'catalog_index': index,
                                'Name': best_row_data['Name']
                            })
                        
                            match_count += 1

                            
                            
                            
                            removed_rows = pd.concat([removed_rows, orders_df[orders_df['Name'] == best_row_data['Name']]])
                            orders_df = orders_df[orders_df['Name'] != best_row_data['Name']]
                            
                            
                        
                        else:
                            break

                    orders_df = pd.concat([orders_df, removed_rows], ignore_index=True)
                    
                    if index % update_interval == 0:  # Update progress every 1000 rows
                        progress_percentage = (index + 1) / num_rows * 100
                        progress_placeholder.write(f"Progress: {progress_percentage:.2f}%")

                matches_df = pd.DataFrame(matches)
                matches_df = matches_df.merge(catalog_df, left_on='catalog_index', right_index=True, how='left')
                matches_df = matches_df.merge(
                    orders_df[['Name', 'discount_code', 'billing_name', 'billing_address1', 'shipping_name', 'shipping_address1']],
                    on='Name', 
                    how='left'
                )
                st.session_state['matches_df'] = matches_df
                catalog_df = matches_df
                st.session_state['catalog_df'] = catalog_df
                st.success("Matching process completed successfully!")
                
            
            
            
            
            
            
           
            
            
            
            
            
            
            
            

            #st.button("Finalize and Download2")

        # Finalize and Download button functionality
        if st.button("Finalize and Download"):
            if 'catalog_df' in st.session_state:
                # Retrieve the cleaned catalog_df from session state
                catalog_df = st.session_state.catalog_df
                
                # Apply transformations and calculations to catalog_df after matching MP_ADD2
                if 'mp_add2' in catalog_df.columns:
                    # Apply transformations and calculations involving 'MP_ADD2'
                    catalog_df['mp_add2'] = catalog_df['mp_add2'].apply(preprocess_text)
                    catalog_df['mp_add2'] = catalog_df['mp_add2'].apply(remove_punctuation)
                    catalog_df['mp_add2'] = catalog_df['mp_add2'].apply(apply_substitutions)
                    catalog_df['mp_add2'] = catalog_df['mp_add2'].apply(preprocess_text)
        
                    catalog_df.loc[:, 'TEST1'] = catalog_df['name'] + ' ' + catalog_df['address'] + ' ' + catalog_df['mp_add2']
                else:
                    # If 'MP_ADD2' doesn't exist, create 'TEST1' without it
                    catalog_df.loc[:, 'TEST1'] = catalog_df['name'] + ' ' + catalog_df['address']

                catalog_df.loc[:, 'TEST2'] = catalog_df['billing_name'] + ' ' + catalog_df['billing_address1']
                catalog_df.loc[:, 'TEST3'] = catalog_df['shipping_name'] + ' ' + catalog_df['shipping_address1']

                catalog_df = catalog_df.dropna(subset=['TEST2'])

                catalog_df.loc[:, 'score1'] = catalog_df.apply(lambda row: fuzz.token_sort_ratio(row['TEST1'], row['TEST2']), axis=1)
                catalog_df.loc[:, 'score2'] = catalog_df.apply(lambda row: fuzz.token_sort_ratio(row['TEST1'], row['TEST3']), axis=1)

                catalog_df.loc[:, 'shipping_street_number_match'] = catalog_df['shipping_street_number_match'] 
                catalog_df.loc[:, 'street_number_match'] = catalog_df['street_number_match'] 

                catalog_df['total_score'] = (
                    catalog_df['name_match'] +
                    catalog_df['billing_address_match'] +
                    catalog_df['shipping_address_match'] +
                    catalog_df['street_number_match'] +
                    catalog_df['shipping_street_number_match'] +
                    catalog_df['score1'] +
                    catalog_df['score2']
                )
                catalog_df['total_score'] = pd.to_numeric(catalog_df['total_score'], errors='coerce')

                catalog_df = catalog_df.loc[catalog_df.groupby('Name')['total_score'].idxmax()]
                catalog_df['match_status'] = 'Unlikely Match'  # Default value

                # Update 'match_status' based on the conditions

                 
                catalog_df.loc[
                    ((catalog_df['score1'] >= 70) & (catalog_df['score1'] < 80)) |
                    ((catalog_df['score2'] >= 70) & (catalog_df['score2'] < 80)) |
                    ((catalog_df['total_score'] > 500) & (catalog_df['total_score'] <= 600)) |
                    (((catalog_df['name_match'] + catalog_df['shipping_address_match'] + catalog_df['shipping_street_number_match']) > 230) &
                    ((catalog_df['name_match'] + catalog_df['shipping_address_match'] + catalog_df['shipping_street_number_match']) <= 260)) |
                    (((catalog_df['name_match'] + catalog_df['billing_address_match'] + catalog_df['billing_address_match']) > 230) &
                    ((catalog_df['name_match'] + catalog_df['billing_address_match'] + catalog_df['billing_address_match']) <= 260)),
                    'match_status'
                ] = 'Possible Match'
                
                catalog_df.loc[
                    (catalog_df['score1'] >= 80) | 
                    (catalog_df['score2'] >= 80) |
                    (catalog_df['total_score'] > 600) |
                    ((catalog_df['shipping_address_match'] + catalog_df['shipping_street_number_match']) > 193) |
                    ((catalog_df['billing_address_match'] + catalog_df['street_number_match']) > 193),
                    'match_status'
                ] = 'Likely Match'
                
                
                result_df = catalog_df.loc[catalog_df.groupby('Name')['total_score'].idxmax()]
                result_df = result_df.sort_values(by='total_score', ascending=False)
                # Provide download link for the final matched catalog DataFrame
                catalog_to_download = result_df
                csv = catalog_to_download.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="catalog_matched.csv">Download matched catalog DataFrame</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
