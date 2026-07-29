from math import pow
from src.utils.utils import *

from src.utils.log_config import logger

class AwarenessSafety:
    def __init__(self, folder_path, awareness_interval_ms=100):
        """
        Object to compute awareness and safety levels based on gaze direction and speed
        """
        self.folder_path = folder_path  # Session folder path
        self.awareness_interval_ms: int = awareness_interval_ms  # Awareness interval in milliseconds
        self.gaze_inside_categories: list = ['INSIDE_CAR', 'SCREEN']  # Define gaze categories inside the car
        self.gaze_undefined_names: list = ['TABLET', 'NONE', 'undefined']  # Define undefined gaze names
        self.awareness_df = pd.DataFrame(columns=['t', 'awareness'])  # Initialize DataFrame to store awareness data
        self.safety_df = pd.DataFrame(columns=['t', 'safety'])  # Initialize DataFrame to store safety data

    def compute_awareness(self):
        """
        Compute awareness level based on gaze direction and streak
        """
        # Check if gaze data is available in the session and load it if not available
        gaze_object = import_dataframe("analytics_gaze_object.csv", self.folder_path)
        if gaze_object is None:
            self.awareness_df = np.nan
            return
        logger.info("Computing Awareness (1/3) GazeObject.csv found in current folder.")

        event_start_ms = 0  # Initialize the start time of an event
        awareness: int = 100  # Initialize awareness level to 100
        buffer_gaze = 0  # Initialize buffer for gaze direction

        gaze_list: list = []  # Initialize a list to store gaze direction over time
        buffer_ms = 0  # Initialize buffer time (time interval between gaze records not processed yet) in milliseconds
        gazed: str = "OUT"  # Initialize gaze direction as 'OUT'
        type_: str = "none"  # Initialize gaze type as 'none'

        # Loop through gaze data
        for r in range(len(gaze_object) - 1):
            record = gaze_object.iloc[r, :]  # Get current gaze record (index r)

            event_start_ms = 0 if r == 0 else event_start_ms + gaze_object.deltaMs[r - 1] + 0  # Set event start time

            # Check if buffer time exceeds awareness interval
            if buffer_ms >= self.awareness_interval_ms:
                if buffer_gaze != 0:
                    gazed = "IN" if buffer_gaze < 0 else "OUT"  # Determine gaze direction based on buffer gaze
                for i in range(buffer_ms // self.awareness_interval_ms):
                    gaze_list.append(gazed)  # Append gaze direction to the list

                buffer_gaze = 0  # Reset buffer gaze
                remaining_time = buffer_ms - (buffer_ms // self.awareness_interval_ms) * self.awareness_interval_ms
                buffer_ms = remaining_time  # Update buffer time with remaining time
                if type_ != "none":
                    buffer_gaze = remaining_time if gazed == "OUT" else -remaining_time  # Update buffer gaze
                logger.debug(f"Buffer gaze: {buffer_gaze}, Buffer ms: {buffer_ms}, Gazed: {gazed}, Type: {type_}")

            # map gaze type
            type_ = 'inside' if (record["category"].strip().upper() in self.gaze_inside_categories) else 'outside'
            # filter out undefined gazes
            if record["gazedName"] in self.gaze_undefined_names:
                type_ = 'undefined'

            if type_ != "none":
                buffer_gaze += record["deltaMs"] if type_ == "outside" else -record["deltaMs"]

            buffer_ms += record["deltaMs"]
        logger.info("Computing Awareness (2/3) Gaze data processed.")

        gaze_streak: int = 0  # Initialize gaze streak counter
        prev_gaze = "NONE"  # Initialize a previous gaze direction as 'NONE'

        # Iterate through the gaze list to calculate awareness
        for idx, g in enumerate(gaze_list):

            if g == prev_gaze or g == "NONE":
                gaze_streak += 1  # Increment streak counter if the gaze direction remains the same
            else:
                gaze_streak = 0  # Reset streak counter if gaze direction changes

            # Update awareness based on gaze direction and streak
            if g == "OUT":
                awareness += pow(1.5, gaze_streak / (1000 / self.awareness_interval_ms))  # if gaze is outside, increase
            if g == "IN":
                awareness -= pow(1.5, gaze_streak / (1000 / self.awareness_interval_ms))  # if gaze is inside, decrease

            if awareness > 100:
                awareness = 100  # Cap awareness level at 100
            elif awareness < 0:
                awareness = 0  # Set awareness level to 0 if it goes below 0

            prev_gaze = prev_gaze if g == "NONE" else g  # Update previous gaze direction

            # Create a new DataFrame with the data for the row
            new_row = pd.DataFrame({'t': [idx * self.awareness_interval_ms], 'awareness': [np.round(awareness, 4)]})
            self.awareness_df = pd.concat([self.awareness_df, new_row], axis=0, ignore_index=True)
        logger.info("Computing Awareness (3/3) Finished!")
        return self.awareness_df.awareness.values.astype('float')

    def compute_safety(self):
        """Compute safety level based on awareness and speed."""

        # Check if gaze data is available in the session and load it if not available
        gaze_object = import_dataframe(os.path.join(self.folder_path, "analytics_gaze_object.csv"), folder_path=self.folder_path, sep=None)
        vehicle_speed = import_dataframe(os.path.join(self.folder_path, "vr_vehicle_speed.csv"), folder_path=self.folder_path, sep=None)
        if gaze_object is None or vehicle_speed is None:
            self.safety_df = np.nan
            return

        if self.awareness_df.empty:
            logger.info("Computing Safety (1/3) Awareness not found in the session. Computing awareness data...")
            self.compute_awareness()

        alpha: int = 1  # Initialize alpha parameter

        # Get maximum speed
        max_speed_record: pd.Series = vehicle_speed.loc[vehicle_speed['VehicleSpeed [m/s]'].idxmax()]
        max_speed: float = max_speed_record['VehicleSpeed [m/s]'] * 3.6  # Convert maximum speed to km/h

        # Iterate through awareness data to calculate safety
        for r in range(self.awareness_df.shape[0] - 1):
            record = self.awareness_df.iloc[r, :]  # Get current awareness record

            event_start_ms = 0 if r == 0 else record['t']

            # Get speed record closest to event start time
            speed_record = vehicle_speed.loc[(vehicle_speed['startEventMs'] - event_start_ms).abs().idxmin()]
            speed = speed_record['VehicleSpeed [m/s]'] * 3.6  # Convert speed to km/h

            # Calculate safety based on awareness and speed
            safety = alpha * record['awareness'] * (1 - (speed / (max_speed * 1.5 if max_speed > 0 else 1.0)))

            safety = 100 if safety > 100 else safety  # Cap safety level at 100

            # Create a new DataFrame with the data for the row
            new_row = pd.DataFrame({'t': record['t'], 'safety': [np.round(safety, 4)]})
            self.safety_df = pd.concat([self.safety_df, new_row], axis=0, ignore_index=True)

        logger.info("Computing Safety (3/3) Finished!")
        return self.safety_df.safety.values.astype('float')