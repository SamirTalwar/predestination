document.addEventListener("DOMContentLoaded", () => {
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

  const socket = io.connect('http://' + document.domain + ':' + location.port)
  socket.on('connect', () => {
    socket.emit('start', {width, height})
  })
  socket.on('generation', newCells => {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (newCells[y][x]) {
          cells[y][x].classList.remove('dead')
          cells[y][x].classList.add('alive')
        } else {
          cells[y][x].classList.remove('alive')
          cells[y][x].classList.add('dead')
        }
      }
    }
  })
})
