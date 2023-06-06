import sqlite3 as sql
import pandas as pd
import os

db_dir = os.getenv('SEND_DB_V3')
db_dir = '/home/daniel.russo/data/send'
send_db_file = os.path.join(db_dir, 'send_3_4.db')

# exhaustive list of domains outlined in
# IG 3.1
domains = [
              'DM',  # demographics
              'SE',  # subject elements
              'CO',  # comments
              'EX',  # exposure
              'DS',  # disposition
              'BW',  # body weight
              'BG',  # body weight gain
              'CL',  # clinical observations
              'DD',  # death diagnosis and detaisl
              'FW',  # food water and consumption
              'LB',  # laboratory test results
              'MA',  # macroscopic findings
              'MI',  # microscopic findings
              'OM',  # organ measurements
              'PM',  # palpable masses
              'PC',  # pharmacokinetics concentrations
              'PP',  # pharmacokinetics parameters
              'SC',  # subjects characteristics
              'TF',  # tumor findings
              'VS',  # vital signs
              'EG',  # ECG test results
              'CV',  # cardiovascular test results
              'RE',  # respiratory test results
              'TE',  # trial elements
              'TX',  # trial sets
              'TA',  # trial arms
              'TS',  # trial summary
              'RELREC', # related records
              'POOLDEF' # pool definition
]

supp_domains = ["SUPP" + dm for dm in domains[:-2]]

all_domains = domains + supp_domains


class SEND:

    def __init__(self, studyid, domains=all_domains):
        self.studyid = studyid
        self.TS = pd.DataFrame()

    def set_domain(self, domain, df: pd.DataFrame):
        setattr(self, domain, df)

    def __repr__(self):
        return self.studyid

    def get_tsparmcd(self, tsparmcd: str):
        return self.TS[self.TS.TSPARMCD == tsparmcd].TSVAL.iloc[0]

    def get_all_subjects_by_arm(self):
        """ returns a a dictionary keyed by the depositor arm with a list of the subjects assigned to that
            arm as the value  """

        arm_assignments = {}

        for armcd, data in self.DM.groupby('ARMCD'):
            arm_assignments[armcd] = data.USUBJID.values.tolist()
        return arm_assignments

    def associate_dose_to_arms(self):
        """ will associate a particular arm with an assigned dose """

        arm_doses = {}

        for setcd, data in self.TX.groupby('SETCD'):
            data.index = data.TXPARMCD

            # some dont have dosage information listed for every arm
            try:
                armcd = data.loc['ARMCD', 'TXVAL']
                arm_doses[armcd] = data.loc['TRTDOS', 'TXVAL']
            except KeyError:
                return None
        return arm_doses

class SENDDB:

    def __init__(self, dbfile):
        self.dbfile = dbfile

    def connect_to_senddb(self):
        """ connects to the gsrs database """
        return sql.connect(self.dbfile)

    def generic_query(self, query_string, query_params=None):
        """ just makes a generic query on the send database """
        conn = self.connect_to_senddb()
        #conn.row_factory = sql.Row
        result = pd.read_sql(query_string, conn, params=query_params)
        conn.close()
        return result

    def generic_execute(self, query_string, query_params=None):
        """ just makes a generic execute on the send database """
        conn = self.connect_to_senddb()
        #conn.row_factory = sql.Row'
        if query_params:
            conn.execute(query_string, query_params)
        else:
            conn.execute(query_string)
        conn.close()

    def get_all_studies_in_domain(self, query_domain='TS'):
        # returns all unique studyid from a particle domain
        # Bo's structure does not have a studies column
        # so there could be different number of unique studies
        # per domain.

        query_string = 'SELECT DISTINCT STUDYID FROM {dm};'.format(dm=query_domain)
        return self.generic_query(query_string)

    def get_all_applications(self):
        return self.generic_query('SELECT DISTINCT APPNUMBER FROM AN')

    def get_studies_by_application(self, app_number):
        return self.generic_query('SELECT STUDYID FROM AN WHERE APPNUMBER=?', (app_number,))

    def table_exists(self, domain):
        query_string = "SELECT name FROM sqlite_master WHERE type='table' AND name='{dm}';".format(dm=domain)
        table_result = self.generic_query(query_string)
        return not table_result.empty

    def get_domains_by_study(self, studyid, query_domains=all_domains):
        send_study = SEND(studyid)

        for dm in query_domains:
            if self.table_exists(dm):
                query_string = "SELECT * from {dm} WHERE STUDYID=?".format(dm=dm)
                df = self.generic_query(query_string, (studyid,))
                if not df.empty:
                    send_study.set_domain(dm, df)

        return send_study

    def get_all_animals(self):
        """ function will get all the animals in DM and impute missing values from
         TS """

        ts_animals = self.generic_query('SELECT STUDYID, TSPARMCD, upper(TSVAL) as TSVAL FROM TS WHERE TSPARMCD == "SPECIES" OR TSPARMCD == "STRAIN"')

        # this is necessary for the pivot
        ts_animals['ID'] = [i for i in range(1, ts_animals.shape[0] + 1)]

        ts_animals['ID'] = ts_animals.groupby(['STUDYID', 'TSPARMCD', 'TSVAL'])['ID'].rank(method='first',
                                                                                           ascending=True).astype(int)

        ts_animals = ts_animals.pivot_table(index=['STUDYID', 'ID'], columns='TSPARMCD', values='TSVAL',
                                            aggfunc='first').reset_index()

        # some studies with multiple
        # strains will wind up having
        # nan values for species.
        ts_animals['SPECIES'] = ts_animals.groupby('STUDYID')['SPECIES'].apply(lambda x: x.ffill().bfill())

        dm = send_db.generic_query('SELECT STUDYID, USUBJID, '
                                   'upper(SPECIES) as SPECIES, '
                                   'upper(STRAIN) as STRAIN, '
                                   'SEX '
                                   'FROM DM')

        animals_merged = ts_animals.merge(dm, on='STUDYID')

        animals_merged.loc[(animals_merged.SPECIES_y.isna()) | (animals_merged.SPECIES_y == ''), 'SPECIES_y'] = \
        animals_merged.loc[(animals_merged.SPECIES_y.isna()) | (animals_merged.SPECIES_y == ''), 'SPECIES_x']
        animals_merged.loc[(animals_merged.STRAIN_y.isna()) | (animals_merged.STRAIN_y == ''), 'STRAIN_y'] = \
        animals_merged.loc[(animals_merged.STRAIN_y.isna()) | (animals_merged.STRAIN_y == ''), 'STRAIN_x']

        animals_merged = animals_merged.drop(['SPECIES_x', 'STRAIN_x'], axis=1).rename({'SPECIES_y': 'SPECIES',
                                                                                        'STRAIN_y': 'STRAIN'}, axis=1)
        return animals_merged


    def get_control_animals(self):
        """
        returns a dataframe of studyid. usbubjid containing control animals
        """

        animals = self.get_all_animals()

        tx = send_db.generic_query("SELECT STUDYID, SETCD, TXVAL FROM TX WHERE TXPARMCD == 'TCNTRL'")
        dm = send_db.generic_query("SELECT STUDYID, USUBJID, SETCD FROM DM")

        animals = animals.merge(dm, on=['STUDYID', 'USUBJID'])
        all_controls = animals.merge(tx, on=['STUDYID', 'SETCD'])

        standAlonesWords = ["placebo", "untreated", "sham"]
        currentModifiers = ["negative", "saline", "peg", "vehicle", "citrate", "dextrose", "water", "air"]
        control_expression = r'|'.join(standAlonesWords + currentModifiers)

        return all_controls[all_controls.TXVAL.str.contains(control_expression, case=False, na=False)]

send_db = SENDDB(send_db_file)

def create_indexes():
    conn = send_db.connect_to_senddb()
    for dm in all_domains:

        q = "SELECT name FROM sqlite_master WHERE type='table' AND name='{dm}';".format(dm=dm)
        table_list = list(conn.execute(q))
        if len(table_list) > 0:
            conn.execute("CREATE INDEX index_{dm} ON {dm} (STUDYID)".format(dm=dm))
            q = "PRAGMA table_info({dm});".format(dm=dm)
            columns = [col for _, col, _, _, _, _ in list(conn.execute(q))]
            if 'USUBJID' in columns:
                conn.execute("CREATE INDEX index_usubjid_{dm} ON {dm} (USUBJID)".format(dm=dm))
    conn.close()

def remove_indexes():
    conn = send_db.connect_to_senddb()
    for dm in all_domains:

        q = "SELECT name FROM sqlite_master WHERE type='table' AND name='{dm}';".format(dm=dm)
        table_list = list(conn.execute(q))
        if len(table_list) > 0:
            conn.execute("DROP INDEX IF EXISTS index_{dm}".format(dm=dm))
            q = "PRAGMA table_info({dm});".format(dm=dm)
            columns = [col for _, col, _, _, _, _ in list(conn.execute(q))]
            if 'USUBJID' in columns:
                conn.execute("DROP INDEX IF EXISTS index_usubjid_{dm}".format(dm=dm))
    conn.close()
