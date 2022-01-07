import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def process(file_name, audioset_wav_df, output_path):
    df = pd.read_csv(file_name)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    data = {}
    wav_csv_df = []
    audioset_wav_df["youtube_id"] = audioset_wav_df["audio_id"].apply(lambda x: x[:11])
    downloaded_yids = set(audioset_wav_df["youtube_id"].values)
    yid_to_filename = dict(zip(audioset_wav_df["youtube_id"], audioset_wav_df["file_name"]))
    audio_id_to_cur_cap_id = {}
    with tqdm(total=df.shape[0]) as pbar:
        for _, row in df.iterrows():
            audio_id = row["youtube_id"]
            if audio_id in downloaded_yids:
                fname = yid_to_filename[audio_id]
                if audio_id not in data:
                    data[audio_id] = {}
                    audio_id_to_cur_cap_id[audio_id] = 0
                    wav_csv_df.append({
                        "audio_id": audio_id,
                        "file_name": fname
                    })
                data[audio_id]["file_name"] = fname
                if "captions" not in data[audio_id]:
                    data[audio_id]["captions"] = []
                audio_id_to_cur_cap_id[audio_id] += 1
                data[audio_id]["captions"].append({
                    "caption": row["caption"], "audiocap_id": row["audiocap_id"],
                    "cap_id": str(audio_id_to_cur_cap_id[audio_id])
                })
            else:
                print(f"Audio file with Youtube id {audio_id} not found")
            pbar.update()
    tmp = data.copy()
    data = { "audios": [] }
    for audio_id, captions in tmp.items():
        item = {"audio_id": audio_id}
        item.update(captions)
        data["audios"].append(item)
    pd.DataFrame(wav_csv_df).to_csv(output_path / "wav.csv", index=False, sep="\t")
    json.dump(data, open(output_path / "text.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audiocaps_annotation", type=str, help="audiocaps annotation directory, containing {train,val,eval}.csv")
    parser.add_argument("audioset_wav_csv", type=str, help="audioset wave file of downloaded clips, csv format, each row '[filename]\t[full_path]'")
    parser.add_argument("--output_path", type=str, default="audiocaps", help="directory to store processed annotation files")
    args = parser.parse_args()

    annotation_path = args.audiocaps_annotation
    output_path = args.output_path
    audioset_wav_csv = args.audioset_wav_csv
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)
    audioset_wav_df = pd.read_csv(audioset_wav_csv, sep="\t")
    for split in ["train", "val", "test"]:
        process(annotation_path / f"{split}.csv", audioset_wav_df, output_path / f"{split}")




