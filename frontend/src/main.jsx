import React from 'react'
import ReactDOM from 'react-dom/client'
import axios from 'axios'
import App from './App.jsx'
import { API_BASE } from './lib/api'
import './index.css'

// Đặt trước khi render để AuthContext và mọi component đều gọi đúng origin.
axios.defaults.baseURL = API_BASE

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
