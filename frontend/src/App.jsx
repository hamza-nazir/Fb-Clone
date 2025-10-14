import { Route, Routes } from "react-router-dom";
import Login from './Pages/Login';
import Signup from './Pages/Signup';
import Home from './Pages/Home';
import Profile from './Pages/Profile'
import { ReduxFun } from "./Hook/Context";
const App = () => {


  return (
    <div>
      <ReduxFun>

      <Routes>
        <Route path="/" element={<Home/> } />
        <Route path="/user/:id" element={<Profile />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />
      </Routes>
      </ReduxFun>
    </div>
  );
};

export default App;
