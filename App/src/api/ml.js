import axios from "axios";

const IP = "http://192.168.1.70:5000";

export default axios.create({
  baseURL: IP,
});
