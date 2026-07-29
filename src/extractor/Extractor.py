import yaml
from matplotlib import pyplot as plt
from scipy import signal as sig
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from src.features import extract_all_features
from src.utils.utils import *
from src.utils.log_config import logger
from .Awareness_safety import AwarenessSafety
from src.app_config.config import *

DATA_PATH = PATH['data']
BASE_PATH = PATH['base_root']
LOGS_JSON_PATH = PATH['logs_json']
MAPS_DIR = os.path.join(os.path.dirname(__file__), '../utils')

_json_map = None


def _get_json_map():
    """Lazily loads the folder -> json mapping (avoids file I/O at import time)."""
    global _json_map
    if _json_map is None:
        map_file_path = os.path.join(BASE_PATH, "data", "map_folder_user.json")
        with open(map_file_path, "r") as f:
            _json_map = json.load(f)
    return _json_map

class MyExtractor:
    """
    A class to analyze the data from the vcockpit dataset.
    """

    def __init__(self, folder_path: str, fs: int = 100, device: str = "Bitalino", save: bool = False):
        self.folder_path = folder_path  # Path to the folder containing the data
        self.user_id = [] 
        self.age = []
        self.gender = []
        self.weight = []
        self.height = []
        self.bmi = []
        self.vr_exp = []
        self.driving_exp = []
        self.driving_freq = []
        self.owns_vehicle = []
        self.vehicle_type = []
        self.nationality = []
        self.drive_configuration = []

        self.fs = fs  # Bitalino PPG signal sampling frequency
        self.device = device
        self.session_start_timestamp, self.session_end_timestamp = extract_session_start_end_timestamp(self.folder_path)
        self.session_duration_ms = self.session_end_timestamp - self.session_start_timestamp
        # Initialize task-related variables -------------------------------------
        self.event_start_scenario = []
        self.event_end_scenario = []
        self.event_start_subject = []
        self.taskIDs = []
        self.event_reaction_time = []
        self.event_execution_time = []
        # Initialize the signals ------------------------------------------------
        self.time_ms = None
        self.time_hr = None
        self.ppg = None
        self.RR = None
        self.hr = None  # Instantaneous HR
        self.eda = None
        self.involvement = None
        self.gaze_through_array = None
        self.category = None
        self.gazed_objects = None
        self.distraction = None
        self.vehicle_speed = None
        self.vehicle_safety_distance_mask = None
        self.distance = None
        self.vehicle_straight_driving_offset = None
        self.lane = None
        self.awareness = None
        self.safety = None
        self.temps_list = None
        self.traffic_light_crossed_red = None
        self.traffic_light_time = None
        self.traffic_light_velocity = None
        # Metadata extraction ---------------------------------------------------
        self.get_task_id()  # Get the task IDs from the vr_scenario_task_sensor.csv file
        self.extract_event()  # Extract event start and end time
        self.scenario = self.get_scenario()
        self.extract_user_data() # Extract user demographic data
        # # Task-related variables extraction --------------------------------------
        self.reaction_execution_time()
        self.traffic_light_check()
        # Initialize the AwarenessSafety object ----------------------------------
        self.AwarenessSafety = AwarenessSafety(self.folder_path)

        logger.info(f">>> Processing session - {os.path.normpath(self.folder_path).split(os.sep)[-1]}")

        if device == "Bitalino" and os.path.exists(os.path.join(self.folder_path, "vr_bitalino.csv")):
            self.extract_ppg() 

            if self.ppg is None or np.all(np.isnan(self.ppg)):
                logger.warning("\tPPG signal is empty or contains only NaN values. Skipping further processing.")
                self.vehicle_speed = []
                self.vehicle_safety_distance_mask = []
                self.distance = []
                self.vehicle_straight_driving_offset = []
                self.lane = []
                self.ppg = []
                self.RR = []
                self.hr = []
                self.involvement = []
                self.gaze_through_array = []
                self.category = []
                self.gazed_objects = []
                self.focus = []
                self.distraction = []
                self.awareness = []
                self.safety = []
                self.folder_paths = [self.folder_path.split('\\')[-1]]
                self.scenario = [self.get_scenario()]
                return 
            
            self.extract_vehicle_speed()
            self.extract_vehicle_safety_distance_mask()
            self.extract_vehicle_straight_driving_variance()
            self.extract_vehicle_lane()
            self.extract_awareness()
            self.extract_safety()
            self.extract_involvement()
            self.extract_focus(smoothing_window_length=31)
        
            self.extract_gaze_track()
            self.extract_distraction()
        else:
            print("'vr_bitalino.csv' not found in current folder. Please check the file path.")

        if save:
            self.save()

    # ---------------------------------------------------------------
    # Overloading the += operator to concatenate two analyzer objects
    # ---------------------------------------------------------------
    def __iadd__(self, other):
        """
        Overload the += operator to concatenate two analyzer objects.
        """

        if not isinstance(other, MyExtractor):
            raise TypeError("Unsupported operand type(s) for +=: 'analyzer' and '{}'".format(type(other)))

        if isinstance(self.vehicle_speed, np.ndarray):
            self.vehicle_speed = [self.vehicle_speed]
            self.distance = [self.distance]
            self.vehicle_safety_distance_mask = [self.vehicle_safety_distance_mask]
            self.vehicle_straight_driving_offset = [self.vehicle_straight_driving_offset]
            self.lane = [self.lane]
            self.ppg = [self.ppg]
            self.RR = [self.RR]
            self.hr = [self.hr]
            self.involvement = [self.involvement]
            self.gaze_through_array = [self.gaze_through_array]
            self.category = [self.category]
            self.gazed_objects = [self.gazed_objects]
            self.focus = [self.focus]
            self.distraction = [self.distraction]
            self.awareness = [self.awareness]
            self.safety = [self.safety]
            self.folder_paths = [self.folder_path.split('\\')[-1]]
            self.scenario = [self.get_scenario()]
    
        # list of lists - Append the new list to the old one
        self.vehicle_speed.append(other.vehicle_speed)
        self.distance.append(other.distance)
        self.vehicle_safety_distance_mask.append(other.vehicle_safety_distance_mask)
        self.vehicle_straight_driving_offset.append(other.vehicle_straight_driving_offset)
        self.lane.append(other.lane)
        self.ppg.append(other.ppg)
        self.RR.append(other.RR)
        self.hr.append(other.hr)
        self.involvement.append(other.involvement)
        self.gaze_through_array.append(other.gaze_through_array)
        self.category.append(other.category)
        self.gazed_objects.append(other.gazed_objects)
        self.focus.append(other.focus)
        self.distraction.append(other.distraction)
        self.awareness.append(other.awareness)
        self.safety.append(other.safety)
        self.folder_paths.append(other.folder_path.split('\\')[-1])
        self.scenario.append(other.scenario)

        # scalars
        if not isinstance(self.session_start_timestamp, list):
            self.session_start_timestamp = [self.session_start_timestamp]
            self.session_end_timestamp = [self.session_end_timestamp]
            self.traffic_light_crossed_red = [self.traffic_light_crossed_red]
            self.traffic_light_time = [self.traffic_light_time]
            self.traffic_light_velocity = [self.traffic_light_velocity]
        
        # append the new scalar to the old one
        self.session_start_timestamp.append(other.session_start_timestamp)
        self.session_end_timestamp.append(other.session_end_timestamp)
        self.event_start_scenario.append(other.event_start_scenario[0])
        self.event_end_scenario.append(other.event_end_scenario[0])
        self.event_start_subject.append(other.event_start_subject)
        self.taskIDs.append(other.taskIDs)
        self.event_reaction_time.append(other.event_reaction_time)
        self.event_execution_time.append(other.event_execution_time)
        self.traffic_light_crossed_red.append(other.traffic_light_crossed_red)
        self.traffic_light_time.append(other.traffic_light_time)
        self.traffic_light_velocity.append(other.traffic_light_velocity)

        # append user data
        self.user_id.append(other.user_id[0])
        self.age.append(other.age[0])
        self.gender.append(other.gender[0])
        self.weight.append(other.weight[0])
        self.height.append(other.height[0])
        self.bmi.append(other.bmi[0])
        self.vr_exp.append(other.vr_exp[0])
        self.driving_exp.append(other.driving_exp[0])
        self.driving_freq.append(other.driving_freq[0])
        self.owns_vehicle.append(other.owns_vehicle[0])
        self.vehicle_type.append(other.vehicle_type[0])
        self.nationality.append(other.nationality[0])
        self.drive_configuration.append(other.drive_configuration[0])

        return self

    # ----------------------
    # Data extraction
    # ----------------------

    def get_scenario(self):
        """
        Get the scenario from the vr_scenario.json file, with "title" key (if it exists, otherwise "tutorial")
        (this is because tutorial sessions do not have the "title" key formatted as the other scenarios).
        """
        scenario = "tutorial"
        json_path = os.path.join(self.folder_path, "vr_scenario.json")
        print(f">>> Loading Scenario from {json_path}")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                json_file = json.load(f)
            if "title" in json_file:
                scenario = json_file["title"]
        logger.info(f"Scenario: {scenario}")
        return scenario
        
    def get_task_id(self):
        """
        Get the task IDs from the vr_scenario_task_sensor.csv file which are contained into 'task_list.yaml'.
        """
        df = import_dataframe("vr_scenario_task_sensor.csv", folder_path=self.folder_path)
        task_list = yaml.load(open(os.path.join(MAPS_DIR, 'task_list.yaml')), Loader=yaml.FullLoader)
        task_list = task_list['taskIDs']
        tasks = list(df.TaskID.replace(' ', '', regex=True))
        # Get task IDs if they are present in the task_list.yaml file
        self.taskIDs.append([task for task in tasks if task in task_list])
        logger.info("Task IDs extracted.")

    def extract_user_data(self):
        """
            Extract user data based on the folder name.
            The mapping between folder_name -> json_file_name
            is stored in map_folder_user.json.
        """
        folder_name = os.path.basename(self.folder_path)
        json_file_name = _get_json_map().get(folder_name)

        if not json_file_name:
            raise ValueError(f"No JSON mapping found for folder: {folder_name}")

        # Path to the user json file
        user_json_path = os.path.join(LOGS_JSON_PATH, json_file_name)

        # Load user data
        with open(user_json_path, "r") as f:
            user_data = json.load(f)

        self.user_id.append(user_data.get('user_id'))

        # extracting demographics from the json file
        self.age.append(user_data.get('age'))
        self.gender.append(user_data.get('gender'))
        self.weight.append(user_data.get('weight_kg'))
        self.height.append(user_data.get('height_cm'))
        self.bmi.append(user_data.get('bmi'))
        self.vr_exp.append(user_data.get('vr_experience_years'))
        self.driving_exp.append(user_data.get('driving_experience_years'))
        self.driving_freq.append(user_data.get('driving_frequency'))
        self.owns_vehicle.append(user_data.get('own_vehicle'))
        self.vehicle_type.append(user_data.get('vehicle_type'))
        self.nationality.append(user_data.get('nationality'))
        self.drive_configuration.append(user_data.get('drive_configuration'))
        
        logger.info("User data extracted.")

    def extract_event(self):
        """
        Extract the start and end time of the events from the vr_scenario_task_sensor.csv file.
        """
        df = import_dataframe("vr_scenario_task_sensor.csv", folder_path=self.folder_path)
        self.event_start_scenario.append(df.startEventMs.values - self.session_start_timestamp)
        self.event_end_scenario.append(df.endEventMs.values - self.session_start_timestamp)
        logger.info("Scenario start and end events timestamps extracted.")

    def reaction_execution_time(self):
        """
        Calculate the reaction time and execution time for a call task.
        """
        # Load the data
        gaze_data = import_dataframe("vr_gaze_detection.csv", folder_path=self.folder_path)
        gaze_data.columns = [col.strip() for col in gaze_data.columns]
        gaze_data.gazedName = gaze_data.gazedName.apply(lambda x: x.strip())
        gaze_data =  gaze_data.loc[gaze_data.gazedName == "SCREEN_Screen_000"]
        gt_data = import_dataframe("vr_scenario_task_sensor.csv", folder_path=self.folder_path)
        gt_data.TaskID = gt_data.TaskID.str.strip()
        gt_data.TaskID = gt_data.TaskID.str.upper()
        # Check if there are tasks into the files
        event_start_subject_temp = []
        for event in self.taskIDs[-1]:
            event = event.strip().upper()
            event_start_scenario = self.event_start_scenario[-1][np.where(gt_data.TaskID == event)[0][0]]
            if 'STOP_CALL' in event:
                logger.info("\tSTOP_CALL event detected.")

                if len(np.where((gaze_data.startEventMs >= event_start_scenario))[0]) == 0:
                    event_start_subject_temp.append(event_start_scenario)
                    print("non trovato")
                else: 
                    event_start_subject_temp.append(gaze_data.startEventMs.iloc[np.where((gaze_data.startEventMs >= event_start_scenario))[0][0]])
            elif 'CALL' in event:
                logger.info("\tCALL event detected.")
                if len(np.where((gaze_data.startEventMs >= event_start_scenario))[0]) == 0:
                    event_start_subject_temp.append(event_start_scenario)
                    print("non trovato")
                else: 
                    event_start_subject_temp.append(gaze_data.startEventMs.iloc[np.where((gaze_data.startEventMs >= event_start_scenario))[0][0]])
            elif 'PLAY' in event:
                logger.info("\tPLAY event detected.")
                event_start_subject_temp.append(gaze_data.startEventMs.iloc[np.where((gaze_data.startEventMs >= event_start_scenario))[0][0]])
            else:
                event_start_subject_temp.append(np.nan)
                continue
        self.event_start_subject.append(event_start_subject_temp)
        self.event_reaction_time.append(self.event_start_subject[-1] - self.event_start_scenario[-1])
        self.event_execution_time.append(self.event_end_scenario[-1] - self.event_start_subject[-1])
        logger.info("Reaction and execution times extracted.")

    def traffic_light_check(self):
        """
        Check if the driver crosses the traffic light respecting the rules.
        """
        df = import_dataframe("vr_trafficsignal_redlight_sensor.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.traffic_light_crossed_red = np.nan
            self.traffic_light_time = np.nan
            self.traffic_light_velocity = np.nan
            return

        self.traffic_light_time = df.endEventMs.values
        self.traffic_light_velocity = df['Velocity[m/s]'].values[0]
        self.traffic_light_crossed_red = 1 if df.TrafficSignalColor.values == " RED" else 0
        logger.info("Traffic light check completed.")

    def extract_ppg(self):
        df = import_dataframe("vr_bitalino.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.ppg = np.nan
            return

        # Initialize the PPG signal and the time array
        new_time = np.arange(0, df.endEventMs.values[-1])
        new_ppg = np.nan * np.ones(len(new_time))

        # Create a new column with absolute time in milliseconds
        time = np.unique(df.startEventMs.values)
        time = np.r_[0, time]
        time = np.c_[time[:-1], time[1:]]

        # Distribute PPG value equally in the considered time interval
        num_rows = time.shape[0]

        for riga in tqdm(range(num_rows), desc="Extracting PPG"):
            # Window length
            w_len = time[riga, 1] - time[riga, 0]
            # Find the corresponding PPG value for the current time interval
            ppg_value = df.query(f"startEventMs > {time[riga, 0]} and endEventMs <= {time[riga, 1]}")["PPG"].values

            if len(ppg_value) == 0:
                continue

            # Window step
            w_step = w_len / len(ppg_value)
            # Distribute ppg_value equally in the considered time interval
            time_interval = (np.arange(time[riga, 0], time[riga, 1], w_step) + w_step).astype(int)
            # Ensure time_interval length matches ppg_value length
            if len(time_interval) > len(ppg_value):
                time_interval = time_interval[:len(ppg_value)]
            elif len(time_interval) < len(ppg_value):
                ppg_value = ppg_value[:len(time_interval)]
            # Assign ppg values to new ppg array
            if riga == 0:
                new_ppg[0:time_interval[0]] = ppg_value[0] * np.ones(time_interval[0])
            elif riga == num_rows - 1:
                time_interval[-1] = time_interval[-1] - 1
                new_ppg[time_interval] = ppg_value
            else:
                new_ppg[time_interval] = ppg_value

        # Makima interpolation to fill nan values
        self.ppg = interp1d(new_time[~np.isnan(new_ppg)], new_ppg[~np.isnan(new_ppg)], kind='cubic', fill_value="extrapolate")(new_time)
        self.time_ms = new_time

    def extract_hr(self):
        """
        Extract the heart rate from the PPG signal.
        """
        x = self.ppg  # Raw ppg values
        x = x - np.mean(x)  # Remove mean
        # signal preprocessing
        y = lowpass_filter(x, self.fs)
        y[y < np.percentile(y, 15)] = np.percentile(y, 15)
        # RR signal extraction
        locs, _ = sig.find_peaks(
            x=y,
            distance=0.5 * self.fs,
            prominence=0.2 * (np.max(y) - np.min(y))
        )
        rr_signal = np.diff(locs)
        rr_seconds = rr_signal / 1000  # RR signal in seconds
        self.RR = rr_seconds  # RR signal (seconds)
        self.hr = 60 / rr_seconds  # Instantaneous Heart rate in bpm (beats per minute)
        self.time_hr = self.time_ms[locs[:-1]]
        logger.info("HR & RR extracted.")
        return x, locs

    def extract_vehicle_speed(self):
        """
        Read the vr_vehicle_speed.csv file and calculate the vehicle speed signal.
        """
        df = import_dataframe("vr_vehicle_speed.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.vehicle_speed = np.nan
            return

        # Interpolate to match the length of the PPG signal
        speed_interp = interpolate_signal_to_ppg_length(signal=df["VehicleSpeed [m/s]"].values,
                                                        ppg_length=len(self.ppg))
        self.vehicle_speed = speed_interp * 3.6  # Convert speed from m/s to km/h
        logger.info("Vehicle speed extracted.")
        return np.array(speed_interp) * 3.6

    def extract_vehicle_safety_distance_mask(self, t_perception_reaction=1.5):
        """
        Calculate the safete distance according to the formula:
        D_total = D_perception_reaction + D_braking =
                = speed * t_perception_reaction + (speed^2) / (2 * mu * g)

        Where mu = friction coefficient (0.7 for dry asphalt)
              g = acceleration due to gravity (9.81 m/s^2)

        We consider a t_perception_reaction of the driver equal to 1.5 seconds according to:
        'Taoka, G. T. J. I. (1989). Break reaction times of unalerted drivers. ITE journal, 59(3), 19-21.'
        """
        df = import_dataframe("vr_distance.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.vehicle_safety_distance_mask = np.nan
            return

        distance = interpolate_signal_to_ppg_length(signal=df["Distance [m]"].values, ppg_length=len(self.ppg))
        self.distance = distance
        speed = self.extract_vehicle_speed()
        # Convert speed from kilometers per hour to meters per second
        speed_mps = speed / 3.6
        # Calculate the distance traveled during the reaction time
        dist_perception_reaction = speed_mps * t_perception_reaction
        # Calculate the braking distance
        dist_braking = (speed_mps ** 2) / (2 * 0.7 * 9.81)
        # Total safety distance is the sum of reaction distance and braking distance
        vehicle_safety_distance = dist_perception_reaction + dist_braking

        # Create a binary mask for safety distance. 1 if the vehicle is within the safety distance, 0 otherwise
        self.vehicle_safety_distance_mask = (distance > vehicle_safety_distance).astype(int)

    def extract_vehicle_straight_driving_variance(self):
        """
        Calculate the variance of the vehicle speed signal during straight driving.
        """
        df = import_dataframe("vr_lane_sensor.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.vehicle_straight_driving_offset = np.nan
            return

        # Interpolate to match the length of the PPG signal
        self.vehicle_straight_driving_offset =interpolate_signal_to_ppg_length(signal=df["Offset [m]"].values,
                                                                                ppg_length=len(self.ppg))

    def extract_vehicle_lane(self):
        """
        Extracts current lane index of the vehicle during the driving session.
        """
        df = import_dataframe("vr_lane_sensor.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.lane = np.nan
            return

        # Interpolate to match the length of the PPG signal
        self.lane =interpolate_signal_to_ppg_length(signal=df["Lane"].values, ppg_length=len(self.ppg))

    def extract_involvement(self):
        """
        Calculate the involvement based on the PPG signal.
        """

        x, locs = self.extract_hr()
        rrn = self.RR[1:]
        rrn1 = self.RR[:-1]
        sd1_teor = np.sqrt(0.5) * np.std(rrn - rrn1)
        sd2_teor = np.sqrt(2 * np.std(rrn) ** 2 - 0.5 * np.std(rrn - rrn1) ** 2)
        xc = np.mean(rrn)
        yc = np.mean(rrn1)
        theta = 45 * np.pi / 180
        p05 = in_ellipse(rrn, rrn1, xc, yc, 0.5 * sd2_teor, 0.5 * sd1_teor, theta).astype(int)
        p1 = in_ellipse(rrn, rrn1, xc, yc, sd2_teor, sd1_teor, theta).astype(int)
        p2 = in_ellipse(rrn, rrn1, xc, yc, 2 * sd2_teor, 2 * sd1_teor, theta).astype(int)
        p3 = in_ellipse(rrn, rrn1, xc, yc, 3 * sd2_teor, 3 * sd1_teor, theta).astype(int)
        p = p05 + p1 + p2 + p3 + 1  # p represent the level of involvement
        # Ensure that x_new is within the bounds of locs
        x_new = np.arange(len(x))
        x_new = np.clip(x_new, locs[1],
                        locs[-2])  # Clip x_new to be within the bounds of locs (excluding the first and last elements)
        p_interpolated = interp1d(locs[1:-1], p, kind='cubic')(x_new)
        
        involvement = pd.Series(p_interpolated).rolling(window=51, min_periods=1).mean()
        # MinMax Scaling involvement array
        self.involvement =np.array(involvement) - min(np.array(involvement)) / (
                max(np.array(involvement)) - min(np.array(involvement)))
        logger.info("Involvement extracted.")

    def extract_focus(self, smoothing_window_length=31):
        """
        Calculate the focus percentage based on the gazed objects.
        """

        # Load the mapping dictionary
        with open(os.path.join(MAPS_DIR, 'FOCUS_entry_mapping.json'), 'r') as file:
            entry_mapping = json.load(file)

        df = import_dataframe("analytics_gaze_object.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.focus = np.nan
            return

        # Create a new column based on the mapping dictionary
        df['gaze_map'] = df['gazedName'].str.strip().map(entry_mapping)
        # Create a new column with absolute time in milliseconds
        df['time_ms'] = np.cumsum(df['deltaMs'])
        # Create two new columns with start and end time of 'gazed object' in milliseconds
        time = df.time_ms.values
        time = np.r_[0, time]
        time = np.c_[time[:-1], time[1:]]
        df['start_time'] = time[:, 0].tolist()
        df['end_time'] = time[:, 1].tolist()
        # Initialize gazed objects and fixation arrays and fill them with mapped gazed array objects segments
        fixation_array = []
        for riga in range(df.shape[0]):
            temp_fixation_array = np.ones(df.loc[riga, "deltaMs"]) * float(df.loc[riga, "gaze_map"])
            if int(df.loc[riga, "deltaMs"]) < 100:
                temp_fixation_array = temp_fixation_array * 2
            fixation_array.extend(temp_fixation_array.tolist())

        # Cropping at the same length of the PPG signal.
        if len(fixation_array) < len(self.ppg):
            fixation_array.extend(np.ones(len(self.ppg) - len(fixation_array)) * fixation_array[-1])
        elif len(fixation_array) > len(self.ppg):
            fixation_array = fixation_array[:len(self.ppg)]
        # During the driving session, the fixation of gazed object is interrupted multiple times. I decide to divide
        # the session in overlapped windows to assess how much time the subject spends in fixation within each window
        window_size = 3 * self.fs
        # Create a view of fixation_array using sliding windows
        windows = sliding_window_view(fixation_array, (window_size,))
        # Calculate the percentage of time spent in fixation for each window
        time_interrupted_fixation = []
        for win in windows:
            time_in_fixation = np.sum((win == 2) | (win == -2))
            percentage_in_fixation = (time_in_fixation / len(win)) * 100
            time_interrupted_fixation.append(percentage_in_fixation)
        time_interrupted_fixation = np.array(time_interrupted_fixation)
        # This array is one "window_length" shorter than the original fixation array (half window at the beginning
        # and the other half at the end). I decide to fill these gaps repeating the nearest values.
        time_interrupted_fixation_ini = np.ones(np.abs(len(time_interrupted_fixation) - len(fixation_array)) // 2) * \
                                        time_interrupted_fixation[0]
        time_interrupted_fixation_fin = np.ones(
            np.abs(len(time_interrupted_fixation) - len(fixation_array)) - len(time_interrupted_fixation_ini)) * \
                                        time_interrupted_fixation[-1]
        time_interrupted_fixation = np.concatenate([time_interrupted_fixation_ini,
                                                    np.array(time_interrupted_fixation),
                                                    time_interrupted_fixation_fin])
        # SCALING - Rescale between 0 and 1 (MinMax) for a better visualization
        time_interrupted_fixation = (100 * (time_interrupted_fixation - np.min(time_interrupted_fixation)) /
                                     (np.max(time_interrupted_fixation) - np.min(time_interrupted_fixation)))
        # SMOOTHING (MOVING AVERAGE)
        self.focus = 100 - np.convolve(time_interrupted_fixation,
                                       np.ones(smoothing_window_length) / smoothing_window_length,
                                       mode='same')
        logger.info("Focus extracted.")

    def extract_gaze_track(self):
        """
        Perform gaze tracking analysis based on the provided gaze data.
        """

        df = import_dataframe("vr_gaze_detection.csv", folder_path=self.folder_path)
        if df is None or df.empty:
            self.gaze_through_array = np.nan
            self.category = np.nan
            self.gazed_objects = np.nan
            return

        # Load the mapping dictionary
        with open(os.path.join(MAPS_DIR, 'GAZE_TRACK_look_through_mapping.json'), 'r') as file:
            look_through_mapping = json.load(file)

        with open(os.path.join(MAPS_DIR, 'GAZE_TRACK_category_mapping.json'), 'r') as file:
            category_mapping = json.load(file)
        
        with open(os.path.join(MAPS_DIR, 'GAZE_TRACK_objects_mapping.json'), 'r') as file:
            gazed_objects_mapping = json.load(file)

        # Create a new column based on the mapping dictionary
        df['look_through_map'] = df['lookThrough'].str.strip().map(look_through_mapping)
        df['category_map'] = df['category'].str.strip().map(category_mapping)
        df['gazed_objects'] = df['gazedName'].str.strip().map(gazed_objects_mapping)
        # Iterate through the columns and remove spaces
        df.columns = [col.strip() for col in df.columns]
        gaze_through_array, category, gazed_objects = [], [], []
        for riga in range(df.shape[0]):
            if riga == 0:
                temp__through_array = np.ones(df.loc[riga, "endEventMs"]) * float(df.loc[riga, "look_through_map"])
                temp_category = np.ones(df.loc[riga, "endEventMs"]) * float(df.loc[riga, "category_map"])
                temp_gazed_obj = np.ones(df.loc[riga, "endEventMs"]) * float(df.loc[riga, "gazed_objects"])
            else:
                temp__through_array = (np.ones(df.loc[riga, "endEventMs"] - df.loc[riga - 1, "endEventMs"]) *
                                       float(df.loc[riga, "look_through_map"]))
                temp_category = (np.ones(df.loc[riga, "endEventMs"] - df.loc[riga - 1, "endEventMs"]) *
                                 float(df.loc[riga, "category_map"]))
                temp_gazed_obj = (np.ones(df.loc[riga, "endEventMs"] - df.loc[riga - 1, "endEventMs"]) *
                                 float(df.loc[riga, "gazed_objects"]))

            gaze_through_array.extend(temp__through_array.tolist())
            category.extend(temp_category.tolist())
            gazed_objects.extend(temp_gazed_obj.tolist())

        # To visualize them correctly, I will extend them considering the last available object gazed.
        if len(gaze_through_array) < len(self.ppg):
            gaze_through_array.extend(np.ones(len(self.ppg) - len(gaze_through_array)) * gaze_through_array[-1])
            category.extend(np.ones(len(self.ppg) - len(category)) * category[-1])
            gazed_objects.extend(np.ones(len(self.ppg) -len(gazed_objects)) * gazed_objects[-1])
        elif len(gaze_through_array) > len(self.ppg):
            gaze_through_array = gaze_through_array[:len(self.ppg)]
            category = category[:len(self.ppg)]
            gazed_objects = gazed_objects[:len(self.ppg)]

        self.gaze_through_array = gaze_through_array
        self.category = category
        self.gazed_objects = gazed_objects
        logger.info("Gaze tracking analysis completed.")

    def extract_distraction_og(self, with_involvement=True, discretize=False):
        """
        Calculate the distraction signal based on the gaze data.
        """
        if self.focus is None:
            self.extract_focus()
        if self.category is None:
            self.extract_gaze_track()

        category_smooth = (np.array(self.category) < 0).astype(int)
        # SMOOTHING (MOVING AVERAGE)
        window_length = 51
        category_smooth = np.convolve(category_smooth, np.ones(window_length) / window_length, mode='same')

        if with_involvement:
            if self.involvement is None:
                self.extract_involvement()
            distraction = (100 - self.focus) * category_smooth * (np.array(self.vehicle_speed) > 0).astype(int) * np.array(self.involvement)
            distraction[distraction > np.percentile(distraction, 97.5)] = np.percentile(distraction,
                                                                                        97.5)
            # SMOOTHING (MOVING AVERAGE)
            window_length = 151
            self.distraction = np.convolve(distraction, np.ones(window_length) / window_length, mode='same')
        else:
            distraction = (100 - self.focus) * category_smooth * (np.array(self.vehicle_speed) > 0).astype(int)
            distraction[distraction > np.percentile(distraction, 97.5)] = np.percentile(distraction, 97.5)
            # SMOOTHING (MOVING AVERAGE)
            window_length = 51
            self.distraction = np.convolve(distraction, np.ones(window_length) / window_length, mode='same')

        if discretize:
            distraction_bin, time_steps = discretize_distraction_signal(self.distraction,
                                                                        window_size=2 * self.fs,
                                                                        sampling_rate=self.fs)

            # SMOOTHING (MOVING AVERAGE)
            window_length = 5
            self.distraction = np.convolve(distraction_bin, np.ones(window_length) / window_length, mode='same')
            logger.info("Distraction extracted.")

    def extract_distraction(self, smooth_win:int = 51, percentile:int = 75, fs=1000):
        if self.gazed_objects is None:
            self.extract_gaze_track()

        object_weights = {
            1: 0.5,     # WINDSCREEN        #POSITIVE
            2: 1.1,     # BUILDING
            3: 1.3,     # CAR               #POSITIVE
            4: 1.5,     # COCKPIT
            5: 1.3,     # DOOR
            6: 0.2,     # LEFT_MIRROR
            7: 0.2,     # REARVIEW MIRROR
            8: 0.2,     # RIGHT_MIRROR
            9: 1.5,     # ROAD              #POSITIVE
            10: 2.5,    # SCREEN_SCREEN_000
            11: 2.5,    # SCREEN_SCREEN_001
            12: 1.0,    # SKYBOX
            13: 0.7,    # STEERING_WHEEL    # POSITIVE
            14: 1.5}    # TABLET

        # 1. Fixation duration in samples → seconds
        durations_samples = self._compute_fixation_durations(self.gazed_objects)
        durations_sec = durations_samples / fs

        # 2. Retrieve object weight per frame
        weights = np.array([
            object_weights.get(int(obj), 0.0) if not np.isnan(obj) else 0.0
            for obj in self.gazed_objects])
        
        # 3a. Exponential time constant
        base_tau = self._estimate_tau(self.gazed_objects, percentile=percentile, fs=fs)
        adjustment = self._compute_tau_adjustement(self.age[0], self.driving_exp[0])
        tau = round(base_tau * (1 + adjustment), 2)

        # 4 Time Factor
        time_factor = np.exp(weights * (durations_sec / tau)) - 1
        # Invert sign of time_factor if gazed_object index in  1, 3, 9 o 13
        negative_indices = {1, 3, 9, 13}
        for i, obj in enumerate(self.gazed_objects):
            if not np.isnan(obj) and int(obj) in negative_indices:
                time_factor[i] = -time_factor[i]

        # 5. Distraction computation
        if len(time_factor) == 0:
            distraction = np.full(time_factor.shape, None, dtype=object)
        else:
            distraction = np.zeros_like(time_factor, dtype=float)
            reference_value = 0
            distraction[0] = time_factor[0]
            
            for i in range(1, len(time_factor)):
                if self.gazed_objects[i] != self.gazed_objects[i-1]:
                    # Gaze changed → reset reference value
                    reference_value = distraction[i-1]
                    distraction_value = reference_value + time_factor[i]
                else:
                    # Accumulate from current reference
                    distraction_value = reference_value + time_factor[i]

                # capping distraction value
                if distraction_value > 100:
                    distraction_value = 100
                elif distraction_value < 0:
                    distraction_value = 0
                distraction[i] = distraction_value

            if smooth_win is not None and smooth_win > 1:
                window = np.ones(smooth_win) / float(smooth_win)
                distraction = np.convolve(distraction, window, mode='same')

            # Ensure numerical bounds
            distraction = np.clip(distraction, 0.0, 100.0)

        self.distraction = np.array(distraction)
        logger.info("Distraction extracted.")

    def extract_awareness(self):
        """Compute awareness level based on awareness and speed"""
        self.awareness = self.AwarenessSafety.compute_awareness()

    def extract_safety(self):
        """Compute safety level based on awareness and speed"""
        self.safety = self.AwarenessSafety.compute_safety()


    def _compute_fixation_durations(self, sequence):
        durations = np.zeros_like(sequence, dtype=float)
        current = 0

        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current += 1
            else:
                current = 0
            durations[i] = current

        return durations

    def _estimate_tau(self, gazed_objects, percentile=75, fs=1000):
        """
        Estimate a good tau (time constant) from the empirical fixation durations.
        Uses the fixation duration 75th percentile (default).
        """
        gazed_objects = np.array(gazed_objects)
        # Compute fixation durations in samples
        durations_samples = self._compute_fixation_durations(gazed_objects)
        durations_sec = durations_samples / fs

        if len(durations_sec) == 0:
            return 0.8  # fallback

        # Pick a high percentile so distraction saturates around typical long fixations
        tau_est = np.percentile(durations_sec, percentile)

        return tau_est

    def _compute_tau_adjustement(self, age, exp, k=0.10):
        """
        Returns a proportional adjustment factor.
        total_score = a + e
        tau_new = tau_base * (1 + adjustment)
        """
        def age_score(age):
            if age > 65:
                return -2
            elif 55 < age <= 65:
                return -1.5
            elif 45 < age <= 55:
                return -1
            elif 35 < age <= 45:
                return 0
            elif 25 < age <= 35:
                return 1
            elif 15 <= age <= 18:
                return 2
            return 0
        
        def experience_score(exp_str):
            """
            exp_str is one of:
            '2-5 years', '5-7 years', '7-10 years', 'more than 10 years'
            """
            if exp_str == "2-5 years":
                return -1
            elif exp_str == "5-7 years":
                return 0
            elif exp_str == "7-10 years":
                return 0
            elif exp_str == "more than 10 years":
                return 1
            return 0

        a = age_score(age)
        e = experience_score(exp)
        total = a + e

        # proportional adjustment
        adjustment = k * total   # can be negative or positive

        return adjustment  # do NOT clamp unless desired

    # ----------------------
    # Feature extraction
    # ----------------------
  
    def extract_features(self, window_length_seconds, overlap: int = 0, mask=None, user=None, scenario=None, output_file_name=None):
        filtered_ppg = butter_filter(self)
        scaled_ppg = robust_scaling(filtered_ppg)

        w_len = window_length_seconds * self.fs
        # Get signal windows
        ppg_windows = reshape_columnwise(scaled_ppg, w_len, overlap)

        window_labels = None
        #reshaping the mask to adapt to the ppg_windows
        if mask is not None:
            step = w_len - overlap
            window_labels = []
            for i in range(ppg_windows.shape[1]):  # Number of windows
                start_idx = i * step
                end_idx = start_idx + w_len
                window_labels.append(1 if np.any(mask[start_idx:end_idx] == 1) else 0)

        if output_file_name is None:
            output_file_name = "features.csv"
        
        extract_all_features(ppg_windows, output_file_name=output_file_name, output_dir=self.folder_path, mask=window_labels, user=user, scenario=scenario)


    # ----------------------
    # Data visualization
    # ----------------------
    def plot_signal(self, signal: str, session=0, x_lim=None, y_lim=None, ax=None):
        """
        Plot the vehicle speed signals.
        """
        # Check if the signal is available
        if signal not in self.signals:
            print(f"Signal not found: {signal}")
            return
        # Time axis to plot the signal
        t = self.time_hr if signal in ["HR", "RR"] else self.time_ms / 1000

        x = self.signals[signal][session] if isinstance(self.signals[signal], list) else self.signals[signal]

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))

        if len(x) == len(self.time_ms):
            ax.plot(t, x)
            ax.set_xlabel("Time [s]", fontsize=16)
            ax.set_ylabel(signal, fontsize=16)
        else:
            ax.plot(x)
            ax.set_xlabel("Time [a.u.]")
            ax.set_ylabel(signal, fontsize=16)

        # Add the event mask to the plot
        if signal == "vehicle_speed":
            mask = self.get_vehicle_safety_distance()
            ax.fill_between(t, min(x), max(x), where=mask, color='green',
                            alpha=0.5)
            ax.legend(["Signal", "Respected safety distance"])
        elif signal in ["ppg", "involvement", "focus", "distraction", "awareness", "safety"]:
            scenario_mask = self.get_event_mask_scenario(session)
            subject_mask = self.get_event_mask_subject(session)
            ax.fill_between(t, min(x), max(x), where=scenario_mask, color='blue',
                            alpha=0.5)
            ax.fill_between(t, min(x), max(x), where=subject_mask, color='red',
                            alpha=0.5)
            ax.legend(["Signal", "Scenario", "Subject"])

        if x_lim is not None:
            ax.xlim(x_lim)
        if y_lim is not None:
            ax.ylim(y_lim)

        if ax is None:
            plt.show()

    def plot_distribution(self, signal: str, session=0):
        """
        Plot the distribution of the signals.
        """

        x = self.signals[signal][session] if isinstance(self.signals[signal], list) else self.signals[signal]

        plt.figure(figsize=(15, 5))
        plt.hist(x, bins=np.sqrt(len(x)))
        plt.xlabel(signal)
        plt.ylabel("Frequency")
        plt.show()

    def plot_all_signals(self, session=0):
        """
        Plot all the signals.
        """
        # Make a figure with number of subplots equal to the number of signals
        fig, axs = plt.subplots(len(self.signals), 1, figsize=(15, 5 * len(self.signals)))
        for i, signal in enumerate(self.signals):
            self.plot_signal(signal, session=session, ax=axs[i])
        plt.tight_layout()
        plt.show()

    # ----------------------
    # Save the analyzer object
    # ----------------------
    def save(self, format_type: str = 'pickle'):
        """
        Save the analyzer object as a pickle file.
        """
        if format_type == 'pickle':
            with open(f"{self.folder_path}_analyzer.pkl", 'wb') as file:
                pickle.dump(self, file)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        logger.info("analyzer object saved.")

    # ----------------------
    # Getters
    # ----------------------
    def get_event_mask_scenario(self, session=0):
        ppg_signal = self.signals["ppg"][session] if isinstance(self.signals["ppg"], list) else self.signals["ppg"]
        mask = np.zeros(len(ppg_signal))
        for i in range(len(self.event_start_scenario[session]) - 1):
            # Skip the "Reach initial speed task" and the last task ("Follow" or "Timeout")
            if i == 0:
                continue
            mask[self.event_start_scenario[session][i]:self.event_end_scenario[session][i]] = 1
        return mask

    def get_event_mask_subject(self, session=0):
        ppg_signal = self.signals["ppg"][session] if isinstance(self.signals["ppg"], list) else self.signals["ppg"]
        mask = np.zeros(len(ppg_signal))
        for i in range(len(self.event_start_subject[session]) - 1):
            if i == 0:
                continue
            mask[self.event_start_subject[session][i]:self.event_end_scenario[session][i]] = 1
        return mask

    # ----------------------
    # Overloading the __str__ method to print the object
    # ----------------------
    def __str__(self):
        return f"Extractor (folder_path={self.folder_path}, fs={self.fs}, device={self.device})"
