document.addEventListener('DOMContentLoaded', () => {
  const CellSize = 16

  const width = Math.floor(window.innerWidth / CellSize)
  const height = Math.floor(window.innerHeight / CellSize)

  const cells = []
  const grid = document.createElement('table')
  grid.classList.add('grid')
  for (let y = 0; y < height; y++) {
    const row = []
    const gridRow = document.createElement('tr')
    for (let x = 0; x < width; x++) {
      const cell = document.createElement('td')
      cell.classList.add('cell', 'dead')
      row.push(cell)
      gridRow.appendChild(cell)
    }
    cells.push(row)
    grid.appendChild(gridRow)
  }
  document.body.appendChild(grid)

  const socket = io('http://' + document.domain + ':' + location.port)
  let state = undefined
  socket.on('connect', () => {
    if (state) {
      socket.emit('next', state)
    } else {
      socket.emit('start', {width, height})
    }
  })
  socket.on('generation', newState => {
    state = newState
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (state[y][x]) {
          cells[y][x].classList.remove('dead')
          cells[y][x].classList.add('alive')
        } else {
          cells[y][x].classList.remove('alive')
          cells[y][x].classList.add('dead')
        }
      }
    }
    setTimeout(() => socket.emit('next', state), 100)
  })
})
